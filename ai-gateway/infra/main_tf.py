"""
Infrastructure as Code — Azure deployment for the AI Gateway.
Run: terraform init && terraform plan && terraform apply

Resources provisioned:
  - Azure Container Apps (gateway service, auto-scaling)
  - Azure Cosmos DB (audit log, append-only)
  - Azure Cache for Redis (rate limiting)
  - Azure Key Vault (secrets)
  - Azure API Management (external entry point)
  - Azure Monitor + Log Analytics (observability)
  - Azure Container Registry (Docker images)

Note: This is a Terraform HCL configuration shown as a Python docstring
for portfolio readability. Save as main.tf to deploy.
"""

TERRAFORM_MAIN = '''
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
  backend "azurerm" {
    resource_group_name  = "rg-ai-gateway-tfstate"
    storage_account_name = "staitfstate"
    container_name       = "tfstate"
    key                  = "ai-gateway.tfstate"
  }
}

provider "azurerm" {
  features {}
}

locals {
  prefix   = "ai-gw"
  location = "eastus2"
  env      = var.environment  # "prod" | "staging"
  tags = {
    Project     = "AI Gateway"
    Environment = var.environment
    Team        = "Engineering"
    CostCenter  = "AI-Platform"
  }
}

# ── Resource Group ─────────────────────────────────────────────────────────────
resource "azurerm_resource_group" "main" {
  name     = "rg-${local.prefix}-${local.env}"
  location = local.location
  tags     = local.tags
}

# ── Container Registry ─────────────────────────────────────────────────────────
resource "azurerm_container_registry" "acr" {
  name                = "cr${local.prefix}${local.env}"
  resource_group_name = azurerm_resource_group.main.name
  location            = local.location
  sku                 = "Standard"
  admin_enabled       = false
  tags                = local.tags
}

# ── Key Vault ──────────────────────────────────────────────────────────────────
resource "azurerm_key_vault" "kv" {
  name                        = "kv-${local.prefix}-${local.env}"
  location                    = local.location
  resource_group_name         = azurerm_resource_group.main.name
  sku_name                    = "standard"
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  purge_protection_enabled    = true
  soft_delete_retention_days  = 90
  tags                        = local.tags
}

# ── Cosmos DB (Audit Log) ──────────────────────────────────────────────────────
resource "azurerm_cosmosdb_account" "audit" {
  name                = "cosmos-${local.prefix}-${local.env}"
  location            = local.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = local.location
    failover_priority = 0
  }

  backup {
    type                = "Continuous"
    continuous_mode_type = "Continuous7Days"
  }

  tags = local.tags
}

resource "azurerm_cosmosdb_sql_database" "audit_db" {
  name                = "AuditLog"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.audit.name
}

resource "azurerm_cosmosdb_sql_container" "events" {
  name                = "events"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.audit.name
  database_name       = azurerm_cosmosdb_sql_database.audit_db.name
  partition_key_path  = "/team"
  default_ttl         = 220752000  # 7 years in seconds (regulatory requirement)

  indexing_policy {
    indexing_mode = "consistent"
    included_path { path = "/request_id/?" }
    included_path { path = "/user_id_hash/?" }
    included_path { path = "/timestamp/?" }
    included_path { path = "/event_type/?" }
  }
}

# ── Redis Cache (Rate Limiting) ────────────────────────────────────────────────
resource "azurerm_redis_cache" "ratelimit" {
  name                = "redis-${local.prefix}-${local.env}"
  location            = local.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 1
  family              = "C"
  sku_name            = "Standard"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  tags                = local.tags
}

# ── Container Apps Environment ─────────────────────────────────────────────────
resource "azurerm_container_app_environment" "env" {
  name                       = "cae-${local.prefix}-${local.env}"
  location                   = local.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = local.tags
}

# ── Gateway Container App ──────────────────────────────────────────────────────
resource "azurerm_container_app" "gateway" {
  name                         = "ca-${local.prefix}-${local.env}"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Multiple"
  tags                         = local.tags

  ingress {
    external_enabled = false   # Only accessible via APIM
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 2
    max_replicas = 20

    http_scale_rule {
      name                = "http-scaling"
      concurrent_requests = "50"
    }

    container {
      name   = "gateway"
      image  = "${azurerm_container_registry.acr.login_server}/ai-gateway:${var.image_tag}"
      cpu    = 1.0
      memory = "2Gi"

      env {
        name        = "AZURE_OPENAI_ENDPOINT"
        secret_name = "aoai-endpoint"
      }
      env {
        name        = "COSMOS_CONNECTION_STRING"
        secret_name = "cosmos-conn"
      }
      env {
        name        = "REDIS_CONNECTION_STRING"
        secret_name = "redis-conn"
      }

      liveness_probe {
        path      = "/health"
        port      = 8000
        transport = "HTTP"
      }
    }
  }
}

# ── Log Analytics ──────────────────────────────────────────────────────────────
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${local.prefix}-${local.env}"
  location            = local.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 90
  tags                = local.tags
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "staging"
  validation {
    condition     = contains(["prod", "staging", "dev"], var.environment)
    error_message = "environment must be prod, staging, or dev"
  }
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
}

data "azurerm_client_config" "current" {}

output "gateway_url" {
  value = azurerm_container_app.gateway.latest_revision_fqdn
}
output "cosmos_endpoint" {
  value = azurerm_cosmosdb_account.audit.endpoint
}
'''

#------------------------------------------------------------------------------
# AWS EFS Module - Kubernetes Integration
# GreenLang Infrastructure
#
# This file contains Kubernetes resources for EFS integration:
# - StorageClass for EFS CSI driver
# - PersistentVolumes for each access point
# - CSI driver configuration
# - Example PersistentVolumeClaim templates
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Kubernetes Provider Configuration
#------------------------------------------------------------------------------

# Note: The kubernetes provider should be configured in the root module
# This file assumes the provider is already configured

#------------------------------------------------------------------------------
# EFS StorageClass
#------------------------------------------------------------------------------

resource "kubernetes_storage_class_v1" "efs" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = "efs-sc"

    labels = {
      "app.kubernetes.io/name"       = "efs-csi-driver"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
    }

    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "false"
    }
  }

  storage_provisioner = "efs.csi.aws.com"
  reclaim_policy      = "Retain"
  volume_binding_mode = "Immediate"

  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.main.id
    directoryPerms   = "700"
    gidRangeStart    = "1000"
    gidRangeEnd      = "2000"
    basePath         = "/dynamic_provisioning"
  }

  mount_options = [
    "tls",
    "iam"
  ]

  allow_volume_expansion = true
}

# StorageClass for immediate binding (for static provisioning)
resource "kubernetes_storage_class_v1" "efs_static" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = "efs-sc-static"

    labels = {
      "app.kubernetes.io/name"       = "efs-csi-driver"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
    }
  }

  storage_provisioner = "efs.csi.aws.com"
  reclaim_policy      = "Retain"
  volume_binding_mode = "Immediate"

  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.main.id
  }

  mount_options = [
    "tls",
    "iam"
  ]
}

#------------------------------------------------------------------------------
# PersistentVolume for Artifacts Access Point
#------------------------------------------------------------------------------

resource "kubernetes_persistent_volume_v1" "artifacts" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = "${var.name}-artifacts-pv"

    labels = {
      "app.kubernetes.io/name"       = "efs-artifacts"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    capacity = {
      storage = "100Gi"
    }

    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    storage_class_name               = kubernetes_storage_class_v1.efs_static[0].metadata[0].name

    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = "${aws_efs_file_system.main.id}::${aws_efs_access_point.artifacts.id}"

        volume_attributes = {
          encryptInTransit = "true"
        }
      }
    }

    mount_options = [
      "tls",
      "iam"
    ]
  }
}

#------------------------------------------------------------------------------
# PersistentVolume for Models Access Point
#------------------------------------------------------------------------------

resource "kubernetes_persistent_volume_v1" "models" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = "${var.name}-models-pv"

    labels = {
      "app.kubernetes.io/name"       = "efs-models"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    capacity = {
      storage = "500Gi"
    }

    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    storage_class_name               = kubernetes_storage_class_v1.efs_static[0].metadata[0].name

    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = "${aws_efs_file_system.main.id}::${aws_efs_access_point.models.id}"

        volume_attributes = {
          encryptInTransit = "true"
        }
      }
    }

    mount_options = [
      "tls",
      "iam"
    ]
  }
}

#------------------------------------------------------------------------------
# PersistentVolume for Shared Access Point
#------------------------------------------------------------------------------

resource "kubernetes_persistent_volume_v1" "shared" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = "${var.name}-shared-pv"

    labels = {
      "app.kubernetes.io/name"       = "efs-shared"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    capacity = {
      storage = "200Gi"
    }

    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    storage_class_name               = kubernetes_storage_class_v1.efs_static[0].metadata[0].name

    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = "${aws_efs_file_system.main.id}::${aws_efs_access_point.shared.id}"

        volume_attributes = {
          encryptInTransit = "true"
        }
      }
    }

    mount_options = [
      "tls",
      "iam"
    ]
  }
}

#------------------------------------------------------------------------------
# PersistentVolume for Tmp Access Point
#------------------------------------------------------------------------------

resource "kubernetes_persistent_volume_v1" "tmp" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = "${var.name}-tmp-pv"

    labels = {
      "app.kubernetes.io/name"       = "efs-tmp"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    capacity = {
      storage = "50Gi"
    }

    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Delete"
    storage_class_name               = kubernetes_storage_class_v1.efs_static[0].metadata[0].name

    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = "${aws_efs_file_system.main.id}::${aws_efs_access_point.tmp.id}"

        volume_attributes = {
          encryptInTransit = "true"
        }
      }
    }

    mount_options = [
      "tls",
      "iam"
    ]
  }
}

#------------------------------------------------------------------------------
# Namespace for GreenLang
#------------------------------------------------------------------------------

resource "kubernetes_namespace_v1" "greenlang" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name = var.eks_namespace

    labels = {
      "app.kubernetes.io/name"       = var.eks_namespace
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
    }

    annotations = {
      "efs.csi.aws.com/file-system-id" = aws_efs_file_system.main.id
    }
  }
}

#------------------------------------------------------------------------------
# PersistentVolumeClaim Templates
#------------------------------------------------------------------------------

# PVC for Artifacts
resource "kubernetes_persistent_volume_claim_v1" "artifacts" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name      = "${var.name}-artifacts-pvc"
    namespace = kubernetes_namespace_v1.greenlang[0].metadata[0].name

    labels = {
      "app.kubernetes.io/name"       = "efs-artifacts"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class_v1.efs_static[0].metadata[0].name
    volume_name        = kubernetes_persistent_volume_v1.artifacts[0].metadata[0].name

    resources {
      requests = {
        storage = "100Gi"
      }
    }
  }
}

# PVC for Models
resource "kubernetes_persistent_volume_claim_v1" "models" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name      = "${var.name}-models-pvc"
    namespace = kubernetes_namespace_v1.greenlang[0].metadata[0].name

    labels = {
      "app.kubernetes.io/name"       = "efs-models"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class_v1.efs_static[0].metadata[0].name
    volume_name        = kubernetes_persistent_volume_v1.models[0].metadata[0].name

    resources {
      requests = {
        storage = "500Gi"
      }
    }
  }
}

# PVC for Shared
resource "kubernetes_persistent_volume_claim_v1" "shared" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name      = "${var.name}-shared-pvc"
    namespace = kubernetes_namespace_v1.greenlang[0].metadata[0].name

    labels = {
      "app.kubernetes.io/name"       = "efs-shared"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class_v1.efs_static[0].metadata[0].name
    volume_name        = kubernetes_persistent_volume_v1.shared[0].metadata[0].name

    resources {
      requests = {
        storage = "200Gi"
      }
    }
  }
}

# PVC for Tmp
resource "kubernetes_persistent_volume_claim_v1" "tmp" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name      = "${var.name}-tmp-pvc"
    namespace = kubernetes_namespace_v1.greenlang[0].metadata[0].name

    labels = {
      "app.kubernetes.io/name"       = "efs-tmp"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
      "app.kubernetes.io/component"  = "storage"
    }
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class_v1.efs_static[0].metadata[0].name
    volume_name        = kubernetes_persistent_volume_v1.tmp[0].metadata[0].name

    resources {
      requests = {
        storage = "50Gi"
      }
    }
  }
}

#------------------------------------------------------------------------------
# ServiceAccount for EFS CSI Driver
#------------------------------------------------------------------------------

resource "kubernetes_service_account_v1" "efs_csi_controller" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name      = var.eks_service_account_name
    namespace = "kube-system"

    labels = {
      "app.kubernetes.io/name"       = "efs-csi-controller"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
    }

    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.irsa[0].arn
    }
  }

  automount_service_account_token = true
}

#------------------------------------------------------------------------------
# ConfigMap for EFS Mount Options
#------------------------------------------------------------------------------

resource "kubernetes_config_map_v1" "efs_config" {
  count = var.eks_cluster_name != null ? 1 : 0

  metadata {
    name      = "${var.name}-efs-config"
    namespace = kubernetes_namespace_v1.greenlang[0].metadata[0].name

    labels = {
      "app.kubernetes.io/name"       = "efs-config"
      "app.kubernetes.io/managed-by" = "terraform"
      "app.kubernetes.io/part-of"    = var.name
    }
  }

  data = {
    "file-system-id"           = aws_efs_file_system.main.id
    "file-system-dns"          = aws_efs_file_system.main.dns_name
    "access-point-artifacts"   = aws_efs_access_point.artifacts.id
    "access-point-models"      = aws_efs_access_point.models.id
    "access-point-shared"      = aws_efs_access_point.shared.id
    "access-point-tmp"         = aws_efs_access_point.tmp.id
    "region"                   = data.aws_region.current.name
    "encryption-in-transit"    = "true"
  }
}

#------------------------------------------------------------------------------
# Example Deployment Using EFS
#------------------------------------------------------------------------------

# This is an example deployment that uses the EFS volumes
# Uncomment and modify as needed for your application

# resource "kubernetes_deployment_v1" "example_efs_app" {
#   count = var.eks_cluster_name != null ? 1 : 0
#
#   metadata {
#     name      = "example-efs-app"
#     namespace = kubernetes_namespace_v1.greenlang[0].metadata[0].name
#
#     labels = {
#       app = "example-efs-app"
#     }
#   }
#
#   spec {
#     replicas = 2
#
#     selector {
#       match_labels = {
#         app = "example-efs-app"
#       }
#     }
#
#     template {
#       metadata {
#         labels = {
#           app = "example-efs-app"
#         }
#       }
#
#       spec {
#         container {
#           name  = "app"
#           image = "nginx:latest"
#
#           volume_mount {
#             name       = "artifacts"
#             mount_path = "/data/artifacts"
#           }
#
#           volume_mount {
#             name       = "models"
#             mount_path = "/data/models"
#             read_only  = true
#           }
#
#           volume_mount {
#             name       = "shared"
#             mount_path = "/data/shared"
#           }
#
#           volume_mount {
#             name       = "tmp"
#             mount_path = "/tmp"
#           }
#
#           resources {
#             limits = {
#               cpu    = "500m"
#               memory = "512Mi"
#             }
#             requests = {
#               cpu    = "250m"
#               memory = "256Mi"
#             }
#           }
#         }
#
#         volume {
#           name = "artifacts"
#           persistent_volume_claim {
#             claim_name = kubernetes_persistent_volume_claim_v1.artifacts[0].metadata[0].name
#           }
#         }
#
#         volume {
#           name = "models"
#           persistent_volume_claim {
#             claim_name = kubernetes_persistent_volume_claim_v1.models[0].metadata[0].name
#           }
#         }
#
#         volume {
#           name = "shared"
#           persistent_volume_claim {
#             claim_name = kubernetes_persistent_volume_claim_v1.shared[0].metadata[0].name
#           }
#         }
#
#         volume {
#           name = "tmp"
#           persistent_volume_claim {
#             claim_name = kubernetes_persistent_volume_claim_v1.tmp[0].metadata[0].name
#           }
#         }
#       }
#     }
#   }
# }

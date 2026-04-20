-- V441__entity_graph.sql
-- Entity Graph: organization → facility → asset → meter hierarchy.
-- v3 L1 Data Foundation product module.
-- Depends on: V001 (extensions).

CREATE TABLE IF NOT EXISTS entity_nodes (
    node_id      TEXT        PRIMARY KEY,
    graph_id     TEXT        NOT NULL,
    node_type    TEXT        NOT NULL,
    name         TEXT        NOT NULL,
    geography    TEXT,
    attributes   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    tenant_id    VARCHAR(64),
    deleted_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_entity_node_type CHECK (node_type IN (
        'organization','facility','asset','meter',
        'supplier','product','activity','emission_source','geography'
    ))
);
CREATE INDEX IF NOT EXISTS idx_entity_node_graph    ON entity_nodes (graph_id);
CREATE INDEX IF NOT EXISTS idx_entity_node_type     ON entity_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_entity_node_tenant   ON entity_nodes (tenant_id);
CREATE INDEX IF NOT EXISTS idx_entity_node_alive
    ON entity_nodes (graph_id) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS entity_edges (
    edge_id      TEXT        PRIMARY KEY,
    graph_id     TEXT        NOT NULL,
    source_id    TEXT        NOT NULL,
    target_id    TEXT        NOT NULL,
    edge_type    TEXT        NOT NULL,
    weight       NUMERIC(12,6) NOT NULL DEFAULT 1.0,
    attributes   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    valid_from   TIMESTAMPTZ,
    valid_to     TIMESTAMPTZ,
    tenant_id    VARCHAR(64),
    deleted_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_entity_edge_type CHECK (edge_type IN (
        'owns','supplies_to','produces','emits',
        'located_in','part_of','consumes','transports'
    ))
);
CREATE INDEX IF NOT EXISTS idx_entity_edge_graph   ON entity_edges (graph_id);
CREATE INDEX IF NOT EXISTS idx_entity_edge_source  ON entity_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_entity_edge_target  ON entity_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_entity_edge_type    ON entity_edges (edge_type);
CREATE INDEX IF NOT EXISTS idx_entity_edge_alive
    ON entity_edges (graph_id) WHERE deleted_at IS NULL;

COMMENT ON TABLE entity_nodes IS
    'Entity Graph v3 nodes (organization/facility/asset/meter hierarchy). '
    'Soft-delete via deleted_at; mirror is greenlang.entity_graph SQLite schema.';
COMMENT ON TABLE entity_edges IS
    'Entity Graph v3 edges (typed directed relationships between nodes).';

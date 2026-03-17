"""PACK-011 SFDR Article 9 Pack configuration module."""


def _write_presets():  # pragma: no cover
    """Helper to generate preset YAML files for Article 9."""
    preset_dir = pathlib.Path(__file__).parent / "presets"
    demo_dir = pathlib.Path(__file__).parent / "demo"
    pack10_preset_dir = pathlib.Path(__file__).parent.parent.parent / "PACK-010-sfdr-article-8" / "config" / "presets"
    pack10_demo_dir = pathlib.Path(__file__).parent.parent.parent / "PACK-010-sfdr-article-8" / "config" / "demo"

    def transform_for_article9(content, preset_name=""):
        """Transform Article 8 preset YAML to Article 9."""
        c = content

        # Replace PACK-010 with PACK-011
        c = c.replace("PACK-010", "PACK-011")

        # Classification
        c = c.replace('sfdr_classification: "ARTICLE_8_PLUS"', 'sfdr_classification: "ARTICLE_9"\narticle9_sub_type: "GENERAL_9_1"')
        c = c.replace('sfdr_classification: "ARTICLE_8"', 'sfdr_classification: "ARTICLE_9"\narticle9_sub_type: "GENERAL_9_1"')

        # PAI - add mandatory flag
        c = c.replace("pai:\n  enabled: true\n  enabled_mandatory_indicators", "pai:\n  enabled: true\n  pai_mandatory: true\n  enabled_mandatory_indicators")

        # Include scope 3
        c = c.replace("  include_scope_3: false", "  include_scope_3: true")

        # Add engagement tracking
        c = c.replace("  data_quality_tracking: true\n\n# --- Taxonomy", "  data_quality_tracking: true\n  engagement_actions_tracking: true\n\n# --- Taxonomy")

        # DNSH - all 18 indicators + strict mode
        c = c.replace("  pai_indicators_used: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]", "  pai_indicators_used: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]")
        c = c.replace("  ungc_violations_exclusion: true\n\n# --- Governance", "  ungc_violations_exclusion: true\n  strict_mode: true\n\n# --- Governance")

        # Add ENGAGEMENT_RECORDS to governance data_sources if not present
        if "ENGAGEMENT_RECORDS" not in c:
            if '    - "PROXY_STATEMENT"\n\n# --- E/S' in c:
                c = c.replace('    - "PROXY_STATEMENT"\n\n# --- E/S', '    - "PROXY_STATEMENT"\n    - "ENGAGEMENT_RECORDS"\n\n# --- E/S')
            elif '    - "ANNUAL_REPORT"\n\n# --- E/S' in c:
                c = c.replace('    - "ANNUAL_REPORT"\n\n# --- E/S', '    - "ANNUAL_REPORT"\n    - "ENGAGEMENT_RECORDS"\n\n# --- E/S')

        # Sustainable investment - enable with 95%
        c = c.replace("sustainable_investment:\n  enabled: false\n  minimum_proportion_pct: 0.0",
            'sustainable_investment:\n  enabled: true\n  minimum_proportion_pct: 95.0\n  require_dnsh_pass: true\n  require_governance_pass: true\n  contribution_assessment_methodology: "IMPACT_BASED"\n  cash_and_hedging_allowance_pct: 5.0\n  continuous_monitoring: true\n  breach_notification_enabled: true')

        # For banks that already have sustainable_investment enabled
        if "sustainable_investment:\n  enabled: true" in c and "minimum_proportion_pct: 20.0" in c:
            c = c.replace("  minimum_proportion_pct: 20.0", "  minimum_proportion_pct: 95.0")
            c = c.replace('  contribution_assessment_methodology: "THRESHOLD_BASED"', '  contribution_assessment_methodology: "IMPACT_BASED"')
            # Add continuous_monitoring and breach_notification
            if "continuous_monitoring" not in c:
                c = c.replace('  contribution_assessment_methodology: "IMPACT_BASED"',
                    '  contribution_assessment_methodology: "IMPACT_BASED"\n  cash_and_hedging_allowance_pct: 5.0\n  continuous_monitoring: true\n  breach_notification_enabled: true')

        # Add Impact, Benchmark, Downgrade sections before Carbon Footprint
        impact_section = '# --- Impact Configuration ---\nimpact:\n  enabled: true\n  primary_objectives:\n    - "CLIMATE_MITIGATION"\n  impact_kpis:\n    - "tonnes_co2_avoided"\n    - "renewable_energy_generated_mwh"\n    - "green_revenue_share_pct"\n  impact_methodology: "CONTRIBUTION_BASED"\n  require_additionality: true\n  theory_of_change_required: true\n  sdg_alignment_tracking: true\n  sdg_targets: [7, 13]\n\n# --- Benchmark Alignment Configuration ---\nbenchmark_alignment:\n  enabled: false\n  benchmark_type: "PAB"\n  explain_no_benchmark: true\n\n# --- Downgrade Monitor Configuration ---\ndowngrade_monitor:\n  enabled: true\n  si_proportion_warning_threshold_pct: 97.0\n  si_proportion_critical_threshold_pct: 95.0\n  monitoring_frequency: "QUARTERLY"\n  remediation_period_days: 30\n\n'
        c = c.replace("# --- Carbon Footprint Configuration ---", impact_section + "# --- Carbon Footprint Configuration ---")

        # Carbon footprint - enable financed emissions and scope 3
        c = c.replace("  calculate_financed_emissions: false", "  calculate_financed_emissions: true")
        if '"SCOPE_3"' not in c.split("# --- Carbon Footprint")[1].split("# --- EET")[0] if "# --- EET" in c else "":
            c = c.replace('    - "SCOPE_2"\n  pcaf_alignment', '    - "SCOPE_2"\n    - "SCOPE_3"\n  pcaf_alignment')
        c = c.replace("  pcaf_alignment: false", "  pcaf_alignment: true")
        c = c.replace("  attribution_analysis: false", "  attribution_analysis: true")

        # EET - add sustainable_investment_fields
        c = c.replace("  pai_consideration_fields: true\n\n# --- Disclosure", "  pai_consideration_fields: true\n  sustainable_investment_fields: true\n\n# --- Disclosure")

        # Disclosure - use Annex III/V for Article 9
        c = c.replace("  annex_ii_enabled: true\n  annex_iii_enabled: true\n  annex_iv_enabled: true",
            "  annex_iii_enabled: true\n  annex_v_enabled: true\n  website_disclosure_enabled: true")
        c = c.replace("  include_asset_allocation_chart: true\n  review_workflow_enabled",
            "  include_asset_allocation_chart: true\n  include_impact_section: true\n  include_benchmark_comparison: true\n  review_workflow_enabled")
        if "  multi_language_support" in c:
            c = c.replace("  include_asset_allocation_chart: true\n  include_impact_section: true\n  include_benchmark_comparison: true\n  multi_language_support",
                "  include_asset_allocation_chart: true\n  include_impact_section: true\n  include_benchmark_comparison: true\n  multi_language_support")
        c = c.replace("  greenwashing_check: true\n\n# --- Screening", "  greenwashing_check: true\n  greenwashing_strict_mode: true\n\n# --- Screening")

        # Screening - expanded exclusions
        if "arctic_drilling" not in c:
            c = c.replace('    - "thermal_coal_extraction"\n  revenue_threshold_pct',
                '    - "thermal_coal_extraction"\n    - "arctic_drilling"\n    - "tar_sands"\n    - "nuclear_weapons"\n    - "cluster_munitions"\n    - "anti_personnel_mines"\n  revenue_threshold_pct')
        else:
            # For bank preset that already has some extra exclusions
            if "nuclear_weapons" not in c:
                c = c.replace('    - "tar_sands"\n  revenue_threshold_pct',
                    '    - "tar_sands"\n    - "nuclear_weapons"\n    - "cluster_munitions"\n    - "anti_personnel_mines"\n  revenue_threshold_pct')

        # Zero-tolerance revenue threshold
        c = c.replace("  revenue_threshold_pct: 5.0", "  revenue_threshold_pct: 0.0")

        # Enable positive screening
        c = c.replace("  positive_screening_enabled: false", "  positive_screening_enabled: true")

        # Add UDHR to norms frameworks
        if "UNIVERSAL_DECLARATION_HUMAN_RIGHTS" not in c:
            c = c.replace('    - "ILO_CONVENTIONS"\n  screening_frequency',
                '    - "ILO_CONVENTIONS"\n    - "UNIVERSAL_DECLARATION_HUMAN_RIGHTS"\n  screening_frequency')
            c = c.replace('    - "ILO_CONVENTIONS"\n  best_in_class',
                '    - "ILO_CONVENTIONS"\n    - "UNIVERSAL_DECLARATION_HUMAN_RIGHTS"\n  best_in_class')
            c = c.replace('    - "OECD_GUIDELINES"\n  screening_frequency',
                '    - "OECD_GUIDELINES"\n    - "UNIVERSAL_DECLARATION_HUMAN_RIGHTS"\n  screening_frequency')

        # Stricter breach handling
        c = c.replace('  breach_handling: "DIVEST_90_DAYS"', '  breach_handling: "IMMEDIATE_DIVEST"')
        c = c.replace('  breach_handling: "ENGAGE_FIRST"', '  breach_handling: "IMMEDIATE_DIVEST"')

        # Reporting - add impact report
        c = c.replace("  include_board_summary: true\n\n# --- Data Quality",
            "  include_board_summary: true\n  include_impact_report: true\n\n# --- Data Quality")

        # Data quality - stricter
        c = c.replace("  require_audited_emissions: false", "  require_audited_emissions: true")
        if "proxy_usage_cap_pct" not in c:
            c = c.replace("  allow_sector_proxies: true\n\n# --- Audit",
                "  allow_sector_proxies: true\n  proxy_usage_cap_pct: 20.0\n\n# --- Audit")

        return c

    # Process each preset
    for preset_name in ["asset_manager", "insurance", "bank", "pension_fund", "wealth_manager"]:
        p10_path = pack10_preset_dir / f"{preset_name}.yaml"
        if p10_path.exists():
            p10_content = p10_path.read_text(encoding="utf-8")
            p11_content = transform_for_article9(p10_content, preset_name)
            (preset_dir / f"{preset_name}.yaml").write_text(p11_content, encoding="utf-8")
            print(f"Created {preset_name}.yaml")
        else:
            print(f"WARNING: {p10_path} not found")

    # Process demo config
    p10_demo = pack10_demo_dir / "demo_config.yaml"
    if p10_demo.exists():
        demo_content = p10_demo.read_text(encoding="utf-8")
        demo_content = demo_content.replace("PACK-010", "PACK-011")
        demo_content = demo_content.replace('sfdr_classification: "ARTICLE_8"', 'sfdr_classification: "ARTICLE_9"\narticle9_sub_type: "GENERAL_9_1"')
        demo_content = demo_content.replace('product_name: "Demo Green Growth Fund"', 'product_name: "Demo Deep Green Impact Fund"')
        demo_content = demo_content.replace("pai:\n  enabled: true\n  enabled_mandatory_indicators", "pai:\n  enabled: true\n  pai_mandatory: true\n  enabled_mandatory_indicators")
        demo_content = demo_content.replace("  include_scope_3: false", "  include_scope_3: true")
        # Sustainable investment enable
        demo_content = demo_content.replace("sustainable_investment:\n  enabled: false\n  minimum_proportion_pct: 0.0",
            'sustainable_investment:\n  enabled: true\n  minimum_proportion_pct: 90.0\n  contribution_assessment_methodology: "IMPACT_BASED"\n  cash_and_hedging_allowance_pct: 10.0\n  continuous_monitoring: false')
        # Disclosure for Article 9
        demo_content = demo_content.replace("  annex_ii_enabled: true\n  annex_iii_enabled: false\n  annex_iv_enabled: true",
            "  annex_iii_enabled: true\n  annex_v_enabled: true\n  website_disclosure_enabled: false")
        # Impact section before carbon footprint
        demo_content = demo_content.replace("# --- Carbon Footprint Configuration ---",
            '# --- Impact Configuration ---\nimpact:\n  enabled: true\n  primary_objectives:\n    - "CLIMATE_MITIGATION"\n  impact_methodology: "CONTRIBUTION_BASED"\n  sdg_alignment_tracking: true\n  sdg_targets: [13]\n\n# --- Downgrade Monitor Configuration ---\ndowngrade_monitor:\n  enabled: false\n\n# --- Carbon Footprint Configuration ---')
        # Add demo-specific fields
        demo_content = demo_content.replace("  mock_mrv_data: true\n  tutorial_mode_enabled",
            "  mock_mrv_data: true\n  mock_impact_data: true\n  tutorial_mode_enabled")

        (demo_dir / "demo_config.yaml").write_text(demo_content, encoding="utf-8")
        print("Created demo_config.yaml")


if __name__ != "__main__":
    pass

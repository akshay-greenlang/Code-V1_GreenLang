# This file contains corrected async test methods - copy these to replace lines 887-1050 in the main test file

    def test_full_async_execution_drying(self, agent, valid_input_drying):
        """Test full async execution with mocked ChatSession for drying process."""
        import asyncio

        async def run_test():
            with patch("greenlang.intelligence.ChatSession") as mock_session_class:
                mock_session = AsyncMock()

                mock_response = Mock()
                mock_response.text = "Textile drying analysis: 150 kW heat demand with 55% solar fraction using evacuated tube collectors."
                mock_response.tool_calls = [
                    {
                        "name": "calculate_process_heat_demand",
                        "arguments": {
                            "process_type": "drying",
                            "production_rate": 500,
                            "temperature_requirement": 120,
                            "inlet_temperature": 20,
                            "specific_heat": 4.18,
                            "latent_heat": 0,
                            "process_efficiency": 0.75,
                            "operating_hours_per_year": 8760,
                        }
                    },
                    {
                        "name": "estimate_solar_thermal_fraction",
                        "arguments": {
                            "process_temperature": 120,
                            "load_profile": "continuous_24x7",
                            "latitude": 28.0,
                            "heat_demand_kw": 150.0,
                            "annual_irradiance": 1900,
                            "storage_hours": 4,
                        }
                    },
                    {
                        "name": "estimate_emissions_baseline",
                        "arguments": {
                            "annual_heat_demand_mwh": 1314.0,
                            "current_fuel_type": "natural_gas",
                            "fuel_efficiency": 0.80,
                        }
                    },
                ]
                mock_response.usage = Mock(cost_usd=0.04, total_tokens=400)
                mock_response.provider_info = Mock(provider="openai", model="gpt-4o-mini")

                mock_session.chat = AsyncMock(return_value=mock_response)
                mock_session_class.return_value = mock_session

                result = await agent._run_async(valid_input_drying)
                return result

        result = asyncio.run(run_test())

        # Verify success
        assert result["success"] is True
        assert "data" in result

        # Verify drying-specific expectations
        data = result["data"]
        assert data["heat_demand_kw"] > 0

        # Higher temperature should result in evacuated tube recommendation
        if "technology_recommendation" in data:
            assert "evacuated" in data["technology_recommendation"].lower() or "flat plate" in data["technology_recommendation"].lower()

    def test_async_execution_with_error_handling(self, agent, valid_input_pasteurization):
        """Test async execution with error handling for budget exceeded."""
        import asyncio

        async def run_test():
            with patch("greenlang.intelligence.ChatSession") as mock_session_class:
                from greenlang.intelligence import BudgetExceeded

                mock_session = AsyncMock()
                mock_session.chat = AsyncMock(side_effect=BudgetExceeded("Budget exceeded: $0.15 > $0.10"))
                mock_session_class.return_value = mock_session

                result = await agent._run_async(valid_input_pasteurization)
                return result

        result = asyncio.run(run_test())

        # Verify error is captured
        assert result["success"] is False
        assert "error" in result
        assert result["error"]["type"] == "BudgetError"
        assert "Budget" in result["error"]["message"]
        assert result["error"]["agent_id"] == agent.agent_id

    def test_async_execution_with_partial_tool_results(self, agent, valid_input_preheating):
        """Test async execution with partial tool results (some tools succeed, some fail)."""
        import asyncio

        async def run_test():
            with patch("greenlang.intelligence.ChatSession") as mock_session_class:
                mock_session = AsyncMock()

                mock_response = Mock()
                mock_response.text = "Partial analysis due to constraints."
                mock_response.tool_calls = [
                    {
                        "name": "calculate_process_heat_demand",
                        "arguments": {
                            "process_type": "preheating",
                            "production_rate": 2000,
                            "temperature_requirement": 180,
                        }
                    },
                    {
                        "name": "estimate_solar_thermal_fraction",
                        "arguments": {
                            "process_temperature": 180,
                            "load_profile": "continuous_24x7",
                            "latitude": 32.0,
                            "heat_demand_kw": 200.0,
                            "annual_irradiance": 1850,
                        }
                    },
                ]
                mock_response.usage = Mock(cost_usd=0.03, total_tokens=300)
                mock_response.provider_info = Mock(provider="openai", model="gpt-4o-mini")

                mock_session.chat = AsyncMock(return_value=mock_response)
                mock_session_class.return_value = mock_session

                result = await agent._run_async(valid_input_preheating)
                return result

        result = asyncio.run(run_test())

        # Should still succeed with available data
        assert result["success"] is True
        data = result["data"]

        # Fields from available tools should be present
        assert data["heat_demand_kw"] >= 0
        assert data["solar_fraction"] >= 0

    def test_build_prompt_generation(self, agent, valid_input_pasteurization):
        """Test _build_prompt generates correct prompt format."""
        operating_hours_per_year = 2920  # 16 hours/day * 5 days/week * 52 weeks

        prompt = agent._build_prompt(valid_input_pasteurization, operating_hours_per_year)

        # Verify prompt contains key sections
        assert "Facility Profile" in prompt
        assert "Location" in prompt
        assert "Requirements" in prompt
        assert "Tasks" in prompt

        # Verify input values are included
        assert "Food & Beverage" in prompt
        assert "pasteurization" in prompt
        assert "1000 kg/hr" in prompt
        assert "72°C" in prompt
        assert "natural_gas" in prompt
        assert "35.0°" in prompt

        # Verify all 7 tool tasks are mentioned
        assert "calculate_process_heat_demand" in prompt
        assert "calculate_temperature_requirements" in prompt
        assert "calculate_energy_intensity" in prompt
        assert "estimate_solar_thermal_fraction" in prompt
        assert "calculate_backup_fuel_requirements" in prompt
        assert "estimate_emissions_baseline" in prompt
        assert "calculate_decarbonization_potential" in prompt

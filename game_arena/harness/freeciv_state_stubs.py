# Copyright 2025 The game_arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stub implementations for FreeCiv game state components used in parsing.

This module provides lightweight stub implementations that replace the use
of Mock objects in production code for natural language parsing.
"""

from typing import Any, Dict, Optional


class FreeCivUnitStub:
    """Minimal unit stub for natural language parsing."""

    def __init__(self, unit_id: int, kind: str):
        self.unit_id = unit_id
        self.kind = kind


class FreeCivCityStub:
    """Minimal city stub for natural language parsing."""

    def __init__(self, city_id: int, name: str):
        self.city_id = city_id
        self.name = name


class FreeCivGameStateStub:
    """Minimal game state stub for natural language parsing.

    This stub provides just enough structure to support natural language
    parsing without requiring a full game state or using Mock objects.
    """

    def __init__(self):
        """Initialize with sample units and cities for ID resolution."""
        # Create sample units for natural language parsing
        self.units: Dict[int, FreeCivUnitStub] = {}
        unit_types = ["settlers", "warrior", "archer", "legion", "phalanx"]

        for i in range(101, 110):
            unit_type = unit_types[(i - 101) % len(unit_types)]
            self.units[i] = FreeCivUnitStub(i, unit_type)

        # Create sample cities for natural language parsing
        self.cities: Dict[int, FreeCivCityStub] = {}
        city_names = ["Rome", "Athens", "Babylon", "Memphis"]

        for i in range(301, 305):
            city_name = city_names[(i - 301) % len(city_names)]
            self.cities[i] = FreeCivCityStub(i, city_name)

    def get_unit_by_id(self, unit_id: int) -> Optional[FreeCivUnitStub]:
        """Get unit by ID if it exists."""
        return self.units.get(unit_id)

    def get_city_by_id(self, city_id: int) -> Optional[FreeCivCityStub]:
        """Get city by ID if it exists."""
        return self.cities.get(city_id)

    def get_unit_by_name(self, name: str) -> Optional[FreeCivUnitStub]:
        """Get first unit matching the given type name."""
        name_lower = name.lower()
        for unit in self.units.values():
            if unit.kind.lower() == name_lower:
                return unit
        return None

    def get_city_by_name(self, name: str) -> Optional[FreeCivCityStub]:
        """Get first city matching the given name."""
        name_lower = name.lower()
        for city in self.cities.values():
            if city.name.lower() == name_lower:
                return city
        return None

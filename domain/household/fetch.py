"""Fetch adaptation of the upstream PARTNR oracle navigation skill."""

from habitat_llm.tools.motor_skills.nav.oracle_nav_skill import OracleNavSkill


class FetchOracleNavSkill(OracleNavSkill):
    def fix_robot_leg(self):
        # The wheeled Fetch has no leg joints to restore during navigation.
        pass

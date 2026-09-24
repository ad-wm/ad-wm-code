"""Local helpers for LeWM's OGBench Scene integration."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _as_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _set_scene_state(original_set_state):
    def set_state(self, qpos, qvel, button_states=None, **kwargs):
        if button_states is not None:
            button_states_arr = np.asarray(button_states).reshape(-1)
            for i in range(min(self._num_buttons, len(button_states_arr))):
                kwargs.setdefault(f"button_state_{i}", button_states_arr[i])

        for i in range(self._num_buttons):
            key = f"button_state_{i}"
            if key in kwargs:
                kwargs[key] = _as_int(kwargs[key])

        return original_set_state(self, qpos, qvel, **kwargs)

    return set_state


def _set_site_axis(model, site_name: str, axis: int, value: float) -> None:
    try:
        model.site(site_name).pos[axis] = value
    except Exception:
        pass


def _set_goal_state(
    self,
    target_pos=None,
    target_quat=None,
    target_button_states=None,
    target_drawer_pos=None,
    target_window_pos=None,
    **kwargs,
):
    """Set a dataset-future-frame Scene goal for LeWM evaluation.

    OGBench Scene success in task mode compares the current cube, button,
    drawer, and window states against target fields stored on the env. LeWM's
    dataset-future-frame protocol therefore needs to populate all of them from
    the selected future row, not just the cube target.
    """

    if target_pos is None:
        target_pos = kwargs.get("goal_privileged_block_0_pos")
    if target_quat is None:
        target_quat = kwargs.get("goal_privileged_block_0_quat")
    if target_button_states is None:
        target_button_states = kwargs.get("goal_button_states")
    if target_drawer_pos is None:
        target_drawer_pos = kwargs.get("goal_privileged_drawer_pos")
    if target_window_pos is None:
        target_window_pos = kwargs.get("goal_privileged_window_pos")

    if target_pos is not None:
        target_quat_arr = None if target_quat is None else np.asarray(target_quat)
        self.set_cube_target_pos(0, np.asarray(target_pos), target_quat_arr)

    if target_button_states is not None:
        states = np.asarray(target_button_states).reshape(-1)
        for i in range(min(self._num_buttons, len(states))):
            self.set_target_button_state(i, int(states[i]))

    for i in range(getattr(self, "_num_buttons", 0)):
        key = f"target_button_state_{i}"
        if key in kwargs:
            self.set_target_button_state(i, _as_int(kwargs[key]))

    if target_drawer_pos is not None:
        drawer_pos = _as_scalar(target_drawer_pos)
        self.set_target_drawer_pos(drawer_pos)
        _set_site_axis(self._model, "drawer_handle_center_target", 1, drawer_pos)

    if target_window_pos is not None:
        window_pos = _as_scalar(target_window_pos)
        self.set_target_window_pos(window_pos)
        _set_site_axis(self._model, "window_handle_center_target", 0, window_pos)

    self._mode = "task"
    try:
        import mujoco

        mujoco.mj_forward(self._model, self._data)
    except Exception:
        pass


def install_scene_goal_state_patch() -> bool:
    """Patch the installed SWM SceneEnv with LeWM Scene eval conveniences."""

    try:
        from stable_worldmodel.envs.ogbench.scene_env import SceneEnv
    except Exception:
        return False

    if not hasattr(SceneEnv, "_lewm_original_set_state"):
        SceneEnv._lewm_original_set_state = SceneEnv.set_state
        SceneEnv.set_state = _set_scene_state(SceneEnv._lewm_original_set_state)

    SceneEnv.set_goal_state = _set_goal_state
    return True


install_scene_goal_state_patch()

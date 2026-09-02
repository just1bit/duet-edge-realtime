from __future__ import annotations

from enum import Enum


class ServiceState(str, Enum):
    STARTING = "starting"
    BUFFERING = "buffering"
    PLAYING = "playing"
    DRAINING = "draining"
    FINISHED = "finished"
    FAILED = "failed"


_TRANSITIONS = {
    ServiceState.STARTING: {ServiceState.BUFFERING, ServiceState.FAILED},
    ServiceState.BUFFERING: {
        ServiceState.PLAYING,
        ServiceState.FINISHED,
        ServiceState.FAILED,
    },
    ServiceState.PLAYING: {ServiceState.DRAINING, ServiceState.FAILED},
    ServiceState.DRAINING: {ServiceState.FINISHED, ServiceState.FAILED},
    ServiceState.FINISHED: set(),
    ServiceState.FAILED: set(),
}


class Lifecycle:
    def __init__(self) -> None:
        self.state = ServiceState.STARTING

    def transition(self, target: ServiceState) -> ServiceState:
        if target == self.state:
            return self.state
        if target not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid service transition {self.state.value} -> {target.value}")
        self.state = target
        return self.state

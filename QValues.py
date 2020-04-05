import numpy as np

Experience = ("Experience", ("state", "reward", "done", "next_state"))


class QValues:

    @staticmethod
    def q_val(dqn, state):
        q_val = dqn.forward(np.array([state]))
        return q_val
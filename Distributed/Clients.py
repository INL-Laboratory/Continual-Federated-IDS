# abstract class
# for each agent, a client is in charge of sending and receiving messages to the controller
import tensorflow as tf
class AgentClient:
    def send_grads(self, msg):
        pass

    def compute_gradient(self, msg): # this function will receive the signal to start computing grad
        pass

    def get_model(self) -> tf.keras.Model:
        # if we aren't using threads and a shared model, the agent should obtain the latest model through it's client
        pass


# abstract class
# each controller uses a broadcaster to communicate with its agents
class Broadcaster:
    def broadcast_to_all(self, msg):  # tells all of the agents to compute gradients (for synchronized version)
        pass

    def broadcast_to_agent(self, id, msg):  # tells a specific agent to compute the gradient
        pass

    def receive(self, msg):
        pass
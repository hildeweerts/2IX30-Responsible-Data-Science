import numpy as np
from scipy.stats import poisson, bernoulli
import copy

class Neighborhood():
    """
    A simulated neighborhood.
    
    Parameters
    ----------
    name : string
        Name of the neighborhood.
    mean_daily : integer
        The average number of daily drug use incidents in this neighborhood.
    historic_discovery_rate : float
        Proportion of actual incidents that have been discovered historically.
    history_length : int
        Number of days of historic data with which we initialize the simulation.
    
        
    Attributes
    ----------
    drug_usage_generator_ : object of class scipy.stats.poisson
        A Poisson discrete random variable that can generate daily drug usage using the rvs() function.
    current_epoch_ : int
        Current epoch in the simulation.
    incidents_ : list of int
        List of number of indicents that have occurred up until current_epoch_.
    discovered_ : list of int
        List of number of indicents that have been discovered up until current_epoch_.
    """
    
    def __init__(self, name, mean_daily, historic_discovery_rate, history_length, seed):
        assert 0 <= historic_discovery_rate <= 1, "historic_discovery_rate must be between 0 and 1"
        
        # set parameters 
        self.name = name
        self.mean_daily = mean_daily
        self.seed = seed 
        
        # initialize drug usage simulation function
        self.drug_usage_generator_ = poisson(mu=mean_daily) # set poisson distribution parameter
        self.drug_usage_generator_.random_state = np.random.RandomState(seed=seed) # set random state for reproducibility
        
        # initialize statistics
        self.current_epoch_ = 0
        self.incidents_ = [self.simulate_incidents() for i in range(history_length)] # simulate incidents for initialization
        self.discovered_ = [int(np.ceil(n*historic_discovery_rate)) for n in self.incidents_] # use historic discovery rate
        self.predicted_ = [-1 for i in range(history_length)]
    
    def simulate_incidents(self):
        """
        Simulate the number of incidents on a given day.
        """
        n_indicents = self.drug_usage_generator_.rvs(size=1)[0]
        return n_indicents
    
    def update(self, n_incidents, n_discovered, n_predicted):
        """
        Update the neighborhood statistics.
        """
        self.current_epoch_ += 1
        self.incidents_.append(n_incidents)
        self.discovered_.append(n_discovered)
        self.predicted_.append(n_predicted)
        
class Model():
    """
    A time series forecasting model that predicts the number of crimes in a region, based on previous incidents.
    """
    def predict(self, discovered):
        """
        Predict the number of incidents based on the number of previously discovered incidents of a neighborhood, 
        by taking the average crime rate.
        
        Parameters
        ----------
        
        discovered : list of int
            list of number of daily discovered incidents for all past days
        """
        prediction = sum(discovered) / len(discovered)
        return prediction
    
class Agent():
    """
    An agent in the simulation who discovers crime. This can be either a police officer or the neighborhood watch.
    
    Parameters
    ----------
    name : string
        The name of the agent.
    agent_type : string
        The type of agent, can be one of ['officer', 'watch']
    discovery_rate : float
        The rate at which the agent discovers crime (a number betweeon 0 and 1).
    neighborhood_name : string
        Optional. If agent is of type 'watch', this parameter indicates to which neighborhood it belongs.
        
    Attributes
    ----------
    discovery_generator_ : object of class scipy.stats.bernoulli
        A bernoulli discrete random variable that can generates discoveries using the rvs() function. 
    """
    def __init__(self, name, agent_type, discovery_rate, seed, neighborhood_name = None):
        # checks
        assert agent_type in ['officer', 'watch'], "Parameter agent_type must be in ['officer', 'watch']"
        if agent_type == 'watch':
            assert neighborhood_name != None, "For agent_type 'watch', neighborhood_name must be provided."
            
        # initialize parameters
        self.name = name
        self.agent_type = agent_type
        self.discovery_rate = discovery_rate
        self.seed = seed
        self.neighborhood_name = neighborhood_name
        
        # initialize discovery generator
        self.discovery_generator_ = bernoulli(p=discovery_rate) # set bernoulli distribution parameter
        self.discovery_generator_.random_state = np.random.RandomState(seed=seed) # set random state for reproducibility
    
    def discover(self, n_incidents):
        """Simulate which incidents are discovered by the agent."""
        discoveries = self.discovery_generator_.rvs(size=n_incidents)
        return discoveries

class Policy():
    """
    Policy for dispatching agents to neighborhoods.
    """
    
    def __init__(self, dispatch_type):
        if dispatch_type == 'basic':
            self.dispatch = self.dispatch_basic
        elif dispatch_type == 'custom':
            self.dispatch = self.dispatch_custom
        else:
            raise ValueError('Invalid dispatch_type')
            
    def dispatch_basic(self, predictions, agents, neighborhoods):
        """Basic dispatching of a police officer and neighborhood watch, based on crime predictions.
        WARNING: this implementation only allows for the simple scenario with two neighborhoods with 
        a neighborhood and one single police officer.
        
        Parameters
        ----------
        predictions : dict
            Dictionary in the form {neighborhood name : predicted number of incidents}
        agents : list of Agent
            The agents that need to be dispatched.
        neighborhoods : list of Neighborhood
            The neighborhoods to which the agents are dispatched.
            
        Returns
        -------
        dispatch_schedule : dict
            Dictionary in the form {Neighborhood.name : Agent}
        """
        # identify neighborhood with highest crime prediction
        hood_max = max(predictions, key=predictions.get)
        
        # create schedule
        dispatch_schedule = {}
        for hood in neighborhoods:
            dispatch_schedule[hood.name] = []
            for agent in agents:
                if (agent.agent_type == 'watch') and (agent.neighborhood_name == hood.name): # dispatch watch to neighborhood
                    dispatch_schedule[hood.name].append(agent)
                elif agent.agent_type == 'officer': # only dispatch if hood has most crime
                    if hood.name == hood_max:
                        dispatch_schedule[hood.name].append(agent)
        return dispatch_schedule
    
    def dispatch_custom(self, predictions):
        """
        You can implement your own custom policy here.
        """
        dispatch_schedule = {}
        return dispatch_schedule
    
class Experiment():
    """
    A simulation experiment.
    
    Parameters
    ----------
    neighborhoods : array-like of Neighborhood
        List of all neighborhoods.
    agents : array-like of Agent
        List of all agents.
    model : Model
        The model that is used to predict the number of incidents in each region.
    policy : Policy
        The policy that is used to dispatch officers based on the predictions for the neighborhoods.
    n_epochs : int
        The number of days the experiment should last.
    """
    def __init__(self, neighborhoods, agents, model, policy, n_epochs):
        self.neighborhoods = copy.deepcopy(neighborhoods)
        self.agents = copy.deepcopy(agents)
        self.model = copy.deepcopy(model)
        self.policy = copy.deepcopy(policy)
        self.n_epochs = n_epochs
        
    def run(self):
        # run experiment
        for epoch in range(self.n_epochs):
            # predict crimes
            predictions = {}
            for hood in self.neighborhoods:
                predictions[hood.name] = self.model.predict(hood.discovered_)
                
            # determine to which neighborhood each agent is dispatched
            dispatched = self.policy.dispatch(predictions=predictions, agents=self.agents, neighborhoods=self.neighborhoods)
            
            # simulate and discover crime
            for hood in self.neighborhoods:
                n_predicted = predictions[hood.name]
                # simulate number of crimes of today
                n_incidents = hood.simulate_incidents()
                # determine which incidents are discovered (note that a single incident can be discovered by multiple agents, it is counted as 1)
                discoveries = np.zeros(n_incidents)
                for agent in dispatched[hood.name]:
                    discoveries += agent.discover(n_incidents)
                n_discovered = np.sum(discoveries > 0)
                # update hood statistics
                hood.update(n_incidents, n_discovered, n_predicted)
                
            # save statistics
            statistics = {}
            for hood in self.neighborhoods:
                statistics[hood.name, 'incidents'] = hood.incidents_
                statistics[hood.name, 'discovered'] = hood.discovered_
                statistics[hood.name, 'predicted'] = hood.predicted_
        return statistics
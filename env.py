import torch
import pandas as pd

class RouteAssignmentEnv:
    def __init__(self, bus_data, berth_capacity=2):
        """
        bus_data: DataFrame of routes
        berth_capacity: int, capacity of each berth
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
        self.bus_data = bus_data
        self.num_buses = len(bus_data)
        self.berth_capacity = berth_capacity
        self.reset()

    def reset(self):
        """
        Resets the environment for a new episode.
        """
        self.berth_1 = []
        self.berth_2 = []
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Returns the current state of the environment.
        """
        return len(self.berth_1), len(self.berth_2)

    '''    
    def assign_buses(self, action):
        """
        Randomly select n routes and assign them to berth 1, 
        the remaining routes go to berth 2.
        """
        n = action
        shuffled_buses = self.bus_data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.berth_1 = shuffled_buses.iloc[:n]['ServiceNo'].tolist()
        self.berth_2 = shuffled_buses.iloc[n:]['ServiceNo'].tolist()
        self.done = True
        return self._get_state()
    '''

    def calculate_delay(self, berth, df):
        """
        Calculates the delay for a given berth.
        """
        total_delay = 0
        queue = []
        time_queue = []
        deck = []
        daycnt = len(df['Date'].unique())
        berth_df = df[df['ServiceNo'].isin(berth)]
        berth_df = berth_df.sort_values(['Date', 'ActualArrival'])

        for index, row in berth_df.iterrows():
            if len(queue) < self.berth_capacity:
                queue.append(row['ServiceNo'])
                time_queue.append(row['ActualArrival'])
                deck.append(row['Type'])

            else:
                delta = row['ActualArrival'] - time_queue[0]
                delta = pd.to_timedelta(delta)
                delta = delta.total_seconds()
                
                # Deck affects service time
                service_time = 40 if row['Type'] == 'SD' else 60

                # Load affects ratio of delay
                if row['Load'] == 'LSD':
                    gamma = 0.8
                elif row['Load'] == 'LDA':
                    gamma = 0.9
                else:
                    gamma = 1.0

                if delta < service_time and delta > 0:  # If day changes, delta < 0
                    total_delay += gamma * (service_time - delta)

                # Next state
                queue.pop(0)
                time_queue.pop(0)
                deck.pop(0)
                queue.append(row['ServiceNo'])
                time_queue.append(row['ActualArrival'])
                deck.append(row['Type'])

        # Average delay per day
        total_delay = total_delay / daycnt

        return torch.tensor(total_delay, device=self.device)

    def calculate_reward(self, alpha, beta, theta, berth_1, berth_2, df):
        """
        Calculates the reward based on the delays and berth distribution.
        """
        delay_berth_1 = self.calculate_delay(berth_1, df)
        delay_berth_2 = self.calculate_delay(berth_2, df)
        reward = -theta * abs(len(self.berth_1) - len(self.berth_2)) - alpha * delay_berth_1 - beta * delay_berth_2
        return reward
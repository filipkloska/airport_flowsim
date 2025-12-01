import mesa
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# --- KONFIGURACJA SYMULACJI ---

MINUTES_PER_STEP = 1
AGENT_SCALING_FACTOR = 10 

# Pojemności stref
CAPACITY_CHECKIN = 200  
CAPACITY_SECURITY = 150
CAPACITY_GATE = 500
CAPACITY_BAGGAGE = 1000
CAPACITY_MAIN_ENTRANCE = 10000

# Czas przebywania w strefie
PROCESS_TIME_CHECKIN = 5
PROCESS_TIME_SECURITY = 5
PROCESS_TIME_BAGGAGE = 10
PROCESS_TIME_GATE = 1 

# --- DANE WEJŚCIOWE ---
raw_data = """
Time      Arr Pax      Dep Pax
00:00      23,707      4,071
01:00      7,529       7,003
02:00      2,006       25,251
03:00      3,542       14,033
04:00      5,046       2,48
05:00      23,274      0,724
06:00      20,354      2,705
07:00      13,065      20,137
08:00      5,634       24,174
09:00      2,68        22,683
10:00      3,21        11,596
11:00      2,661       2,164
12:00      15,192      6,752
13:00      12,836      4,324
14:00      8,421       19,609
15:00      4,204       11,016
16:00      4,045       5,927
17:00      6,763       5,775
18:00      10,501      5,849
19:00      12,365      5,222
20:00      8,51        8,451
21:00      7,237       8,539
22:00      11,289      5,364
23:00      15,816      5,786
"""

def parse_schedule(raw_data):
    rows = []
    lines = raw_data.strip().split('\n')
    for line in lines[1:]:
        parts = line.split()
        time_str = parts[0]
        arr_pax = float(parts[1].replace(',', '.')) * 1000
        dep_pax = float(parts[2].replace(',', '.')) * 1000
        
        rows.append({
            "hour": int(time_str.split(':')[0]),
            "arr_agents": int(arr_pax / AGENT_SCALING_FACTOR),
            "dep_agents": int(dep_pax / AGENT_SCALING_FACTOR)
        })
    return pd.DataFrame(rows)

schedule_df = parse_schedule(raw_data)

# --- KLASY MODELU ---

class AirportAgent(mesa.Agent):
    """Agent reprezentujący pasażera."""
    def __init__(self, unique_id, model, agent_type, start_node):
        # POPRAWKA: Inicjalizacja agenta zgodna z nową wersją Mesa
        super().__init__(model)
        self.unique_id = unique_id
        self.agent_type = agent_type
        self.current_node = start_node
        self.process_timer = 0
        
        if self.agent_type == 'departure':
            self.path_sequence = ['main_entrance', 'checkin', 'security', 'gate_dep', 'exit']
        else:
            self.path_sequence = ['gate_arr', 'baggage', 'main_entrance', 'exit']
            
    def get_node_type(self, node_id):
        return self.model.graph.nodes[node_id]['type']

    def step(self):
        if self.process_timer > 0:
            self.process_timer -= 1
            return

        current_type = self.get_node_type(self.current_node)
        
        try:
            current_seq_idx = self.path_sequence.index(current_type)
        except ValueError:
            self.model.schedule.remove(self)
            return

        next_type = self.path_sequence[current_seq_idx + 1]
        
        if next_type == 'exit':
            self.model.remove_agent_from_node(self, self.current_node)
            self.model.schedule.remove(self)
            return

        neighbors = list(self.model.graph.neighbors(self.current_node))
        candidates = [n for n in neighbors if self.model.graph.nodes[n]['type'] == next_type]
        
        if not candidates:
            return

        # Load Balancing
        candidates.sort(key=lambda n: self.model.get_occupancy(n))
        best_node = candidates[0]

        node_capacity = self.model.graph.nodes[best_node]['capacity']
        current_occupancy = len(self.model.grid.get_cell_list_contents([best_node]))
        
        if current_occupancy < node_capacity:
            self.model.move_agent(self, self.current_node, best_node)
            self.current_node = best_node
            
            if next_type == 'checkin': self.process_timer = PROCESS_TIME_CHECKIN
            elif next_type == 'security': self.process_timer = PROCESS_TIME_SECURITY
            elif next_type == 'baggage': self.process_timer = PROCESS_TIME_BAGGAGE
            elif 'gate' in next_type: self.process_timer = PROCESS_TIME_GATE

class AirportModel(mesa.Model):
    def __init__(self, schedule_data):
        # POPRAWKA 1: super().__init__() tworzy self.random
        super().__init__() 
        
        self.schedule_data = schedule_data
        self.current_step = 0
        self.current_hour = 0
        
        # POPRAWKA 2: Ręczny licznik ID (next_id usunięto z Mesa 3.0)
        self._current_id = 0
        
        self.schedule = mesa.time.RandomActivation(self)
        
        # POPRAWKA 3: Najpierw budujemy graf, potem podpinamy Grid
        # (Zapobiega KeyError: 'agent' przy inicjalizacji)
        self.graph = nx.DiGraph()
        self.build_airport_graph()
        self.grid = mesa.space.NetworkGrid(self.graph)
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "CheckIn_Queue": lambda m: m.get_total_occupancy('checkin'),
                "Security_Queue": lambda m: m.get_total_occupancy('security'),
                "MainEntrance_Congestion": lambda m: m.get_total_occupancy('main_entrance'),
                "Baggage_Queue": lambda m: m.get_total_occupancy('baggage'),
                "Dep_Gates_Occupancy": lambda m: m.get_total_occupancy('gate_dep'),
                "Arr_Gates_Occupancy": lambda m: m.get_total_occupancy('gate_arr')
            }
        )

    def next_id(self):
        """Metoda pomocnicza dla generowania ID."""
        self._current_id += 1
        return self._current_id

    def build_airport_graph(self):
        self.graph.add_node("ME", type="main_entrance", capacity=CAPACITY_MAIN_ENTRANCE)
        
        checkins = [f"CI_{i}" for i in range(20)]
        for ci in checkins:
            self.graph.add_node(ci, type="checkin", capacity=CAPACITY_CHECKIN)
            
        securities = [f"SEC_{i}" for i in range(10)]
        for sec in securities:
            self.graph.add_node(sec, type="security", capacity=CAPACITY_SECURITY)
            
        dep_gates = [f"G_DEP_{i}" for i in range(10)]
        arr_gates = [f"G_ARR_{i}" for i in range(10)]
        for g in dep_gates: self.graph.add_node(g, type="gate_dep", capacity=CAPACITY_GATE)
        for g in arr_gates: self.graph.add_node(g, type="gate_arr", capacity=CAPACITY_GATE)
        
        self.graph.add_node("BC", type="baggage", capacity=CAPACITY_BAGGAGE)

        for ci in checkins:
            self.graph.add_edge("ME", ci)
        for ci in checkins:
            for sec in securities:
                self.graph.add_edge(ci, sec)
        for sec in securities:
            for g in dep_gates:
                self.graph.add_edge(sec, g)
        for g in arr_gates:
            self.graph.add_edge(g, "BC")
        self.graph.add_edge("BC", "ME")

    def get_occupancy(self, node):
        count = len(self.grid.get_cell_list_contents([node]))
        cap = self.graph.nodes[node]['capacity']
        return count / cap if cap > 0 else 1

    def get_total_occupancy(self, node_type):
        total = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] == node_type:
                total += len(self.grid.get_cell_list_contents([node]))
        return total

    def move_agent(self, agent, old_node, new_node):
        self.grid.move_agent(agent, new_node)

    def remove_agent_from_node(self, agent, node):
        self.grid.remove_agent(agent)

    def spawn_agents(self):
        row = self.schedule_data[self.schedule_data['hour'] == self.current_hour]
        if row.empty: return

        arr_count = row.iloc[0]['arr_agents']
        dep_count = row.iloc[0]['dep_agents']

        # POPRAWKA 4: Użycie np.random zamiast self.random dla rozkładu Poissona
        agents_per_min_dep = dep_count / 60.0
        num_new_dep = int(np.random.poisson(agents_per_min_dep))
        
        for _ in range(num_new_dep):
            a = AirportAgent(self.next_id(), self, 'departure', 'ME')
            self.schedule.add(a)
            self.grid.place_agent(a, 'ME')

        agents_per_min_arr = arr_count / 60.0
        num_new_arr = int(np.random.poisson(agents_per_min_arr))
        
        arr_gates = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == 'gate_arr']
        
        for _ in range(num_new_arr):
            arr_gates.sort(key=lambda n: self.get_occupancy(n))
            start_gate = arr_gates[0]
            
            a = AirportAgent(self.next_id(), self, 'arrival', start_gate)
            self.schedule.add(a)
            self.grid.place_agent(a, start_gate)

    def step(self):
        minutes_total = self.current_step * MINUTES_PER_STEP
        self.current_hour = (minutes_total // 60) % 24
        
        self.spawn_agents()
        self.schedule.step()
        self.datacollector.collect(self)
        self.current_step += 1

# --- URUCHOMIENIE ---

if __name__ == "__main__":
    print("Rozpoczynanie symulacji lotniska (24h)...")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    model = AirportModel(schedule_df)

    total_steps = 24 * 60 

    for i in range(total_steps):
        model.step()
        if i % 60 == 0:
            print(f"Symulacja: Godzina {model.current_hour}:00 zakończona.")

    # --- ANALIZA WYNIKÓW ---
    results = model.datacollector.get_model_vars_dataframe()

    plt.figure(figsize=(14, 8))
    time_hours = results.index / 60.0

    plt.plot(time_hours, results["MainEntrance_Congestion"], label="Main Entrance", alpha=0.7)
    plt.plot(time_hours, results["CheckIn_Queue"], label="Check-In", linewidth=2)
    plt.plot(time_hours, results["Security_Queue"], label="Security", linewidth=2)
    plt.plot(time_hours, results["Baggage_Queue"], label="Baggage", linewidth=2)
    plt.plot(time_hours, results["Dep_Gates_Occupancy"], label="Dep Gates", linestyle='--')
    plt.plot(time_hours, results["Arr_Gates_Occupancy"], label="Arr Gates", linestyle='--')

    plt.title("Analiza Wąskich Gardeł Lotniska (24h)")
    plt.xlabel("Godzina symulacji")
    plt.ylabel("Liczba Agentów")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, 25))
    plt.tight_layout()
    plt.show()
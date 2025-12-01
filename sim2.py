import mesa
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
import sys
import argparse
import math

# --- KONFIGURACJA SYMULACJI ---

MINUTES_PER_STEP = 1
AGENT_SCALING_FACTOR = 10 

# Limity fizyczne lotniska
TOTAL_CHECKINS = 20
TOTAL_SECURITIES = 10
TOTAL_GATES_DEP = 10
TOTAL_GATES_ARR = 10

# Pojemności stref
CAPACITY_CHECKIN = 200  
CAPACITY_SECURITY = 150
CAPACITY_GATE = 500
CAPACITY_BAGGAGE = 1000
CAPACITY_MAIN_ENTRANCE = 10000

# Czas przebywania w strefie (minuty)
PROCESS_TIME_CHECKIN = 5
PROCESS_TIME_SECURITY = 5
PROCESS_TIME_BAGGAGE = 10
PROCESS_TIME_GATE = 30 

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
    def __init__(self, unique_id, model, agent_type, start_node):
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
            self.model.add_throughput(current_type)
            return

        neighbors = list(self.model.graph.neighbors(self.current_node))
        
        candidates = [
            n for n in neighbors 
            if self.model.graph.nodes[n]['type'] == next_type 
            and self.model.graph.nodes[n]['active']
        ]
        
        if not candidates:
            return

        candidates.sort(key=lambda n: self.model.get_occupancy(n))
        best_node = candidates[0]

        node_capacity = self.model.graph.nodes[best_node]['capacity']
        current_occupancy = len(self.model.grid.get_cell_list_contents([best_node]))
        
        if current_occupancy < node_capacity:
            self.model.move_agent(self, self.current_node, best_node)
            self.current_node = best_node
            self.model.add_throughput(current_type)
            
            if next_type == 'checkin': self.process_timer = PROCESS_TIME_CHECKIN
            elif next_type == 'security': self.process_timer = PROCESS_TIME_SECURITY
            elif next_type == 'baggage': self.process_timer = PROCESS_TIME_BAGGAGE
            elif 'gate' in next_type: self.process_timer = PROCESS_TIME_GATE

class AirportModel(mesa.Model):
    def __init__(self, schedule_data, config=None):
        super().__init__() 
        
        self.schedule_data = schedule_data
        self.current_step = 0
        self.current_hour = 0
        self._current_id = 0
        
        self.config = config if config else {}
        self.adaptive_mode = self.config.get('adaptive', False)
        self.optimized_mode = self.config.get('optimized', False)
        self.queue_tolerance = 50 
        
        self.throughput_counters = {
            'checkin': 0, 'security': 0, 'gate_dep': 0, 'gate_arr': 0, 'baggage': 0
        }
        
        self.schedule = mesa.time.RandomActivation(self)
        
        self.graph = nx.DiGraph()
        self.build_airport_graph()
        self.grid = mesa.space.NetworkGrid(self.graph)
        
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "CheckIn_Queue": lambda m: m.get_total_occupancy('checkin'),
                "Security_Queue": lambda m: m.get_total_occupancy('security'),
                "Active_CheckIns": lambda m: m.count_active_nodes('checkin'),
                "Active_Security": lambda m: m.count_active_nodes('security'),
                "Flow_CheckIn_Cum": lambda m: m.throughput_counters['checkin'],
                "Flow_Security_Cum": lambda m: m.throughput_counters['security'],
                "Flow_GateDep_Cum": lambda m: m.throughput_counters['gate_dep'],
                "Flow_GateArr_Cum": lambda m: m.throughput_counters['gate_arr'],
                "Flow_Baggage_Cum": lambda m: m.throughput_counters['baggage']
            }
        )

    def next_id(self):
        self._current_id += 1
        return self._current_id

    def add_throughput(self, node_type):
        if node_type in self.throughput_counters:
            self.throughput_counters[node_type] += 1

    def build_airport_graph(self):
        self.graph.add_node("ME", type="main_entrance", capacity=CAPACITY_MAIN_ENTRANCE, active=True)
        
        for i in range(TOTAL_CHECKINS):
            self.graph.add_node(f"CI_{i}", type="checkin", capacity=CAPACITY_CHECKIN, active=True)
        for i in range(TOTAL_SECURITIES):
            self.graph.add_node(f"SEC_{i}", type="security", capacity=CAPACITY_SECURITY, active=True)
        for i in range(TOTAL_GATES_DEP):
            self.graph.add_node(f"G_DEP_{i}", type="gate_dep", capacity=CAPACITY_GATE, active=True)
        for i in range(TOTAL_GATES_ARR):
            self.graph.add_node(f"G_ARR_{i}", type="gate_arr", capacity=CAPACITY_GATE, active=True)
        
        self.graph.add_node("BC", type="baggage", capacity=CAPACITY_BAGGAGE, active=True)

        checkins = [n for n in self.graph.nodes if "CI_" in n]
        securities = [n for n in self.graph.nodes if "SEC_" in n]
        dep_gates = [n for n in self.graph.nodes if "G_DEP_" in n]
        arr_gates = [n for n in self.graph.nodes if "G_ARR_" in n]

        for ci in checkins: self.graph.add_edge("ME", ci)
        for ci in checkins:
            for sec in securities: self.graph.add_edge(ci, sec)
        for sec in securities:
            for g in dep_gates: self.graph.add_edge(sec, g)
        for g in arr_gates: self.graph.add_edge(g, "BC")
        self.graph.add_edge("BC", "ME")

    def update_staffing(self):
        """Strategia alokacji zasobów."""
        if not self.adaptive_mode: return

        row = self.schedule_data[self.schedule_data['hour'] == self.current_hour]
        if row.empty: return
        
        dep_agents = row.iloc[0]['dep_agents'] 
        throughput_checkin = 60 / PROCESS_TIME_CHECKIN
        throughput_security = 60 / PROCESS_TIME_SECURITY 
        
        raw_checkin_needed = dep_agents / throughput_checkin
        raw_security_needed = dep_agents / throughput_security

        if self.optimized_mode:
            target_checkin = math.floor(raw_checkin_needed)
            target_security = math.floor(raw_security_needed)
            
            if self.get_total_occupancy('checkin') > self.queue_tolerance: target_checkin += 1
            if self.get_total_occupancy('security') > self.queue_tolerance: target_security += 1
        else:
            target_checkin = math.ceil(raw_checkin_needed)
            target_security = math.ceil(raw_security_needed)

        active_checkins = max(1, min(target_checkin, TOTAL_CHECKINS))
        active_securities = max(1, min(target_security, TOTAL_SECURITIES))
        
        self._set_active_nodes('checkin', active_checkins)
        self._set_active_nodes('security', active_securities)

    def _set_active_nodes(self, node_type, count):
        nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == node_type]
        nodes.sort()
        for i, node in enumerate(nodes):
            # WAŻNE: Nie nadpisuj awarii (awaria ma priorytet nad staffingiem)
            # Tutaj ustawiamy tylko intencję otwarcia. 
            # Rzeczywista dostępność (active) jest filtrowana w trigger_failures.
            # Dla uproszczenia w tym modelu: staffing ustawia bazę, awarie psują bazę.
            self.graph.nodes[node]['active'] = (i < count)

    def trigger_failures(self):
        """Aplikowanie scenariuszy wydarzeń."""
        # 1. Najpierw ustawiamy stan wynikający ze strategii (lub resetujemy do full)
        if self.adaptive_mode:
            self.update_staffing()
        else:
            for n in self.graph.nodes: self.graph.nodes[n]['active'] = True
            
        # 2. Aplikujemy awarie LOSOWE
        sec_prob = self.config.get('security_prob', 0.0)
        gate_prob = self.config.get('gate_prob', 0.0)

        # Pobieramy te, które strategia chciała mieć otwarte
        active_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['active']]
        for n in active_nodes:
            ntype = self.graph.nodes[n]['type']
            prob = sec_prob if ntype == 'security' else (gate_prob if ntype == 'gate_dep' else 0)
            if prob > 0 and random.random() < prob:
                self.graph.nodes[n]['active'] = False
                
        # 3. Aplikujemy awarie PLANOWANE (Blackouts)
        scheduled = self.config.get('scheduled_downtime', {})
        for node_type, (start, duration) in scheduled.items():
            if start <= self.current_hour < start + duration:
                 target_nodes = [n for n in self.graph.nodes if self.graph.nodes[n]['type'] == node_type]
                 for node in target_nodes: self.graph.nodes[node]['active'] = False

    def count_active_nodes(self, node_type):
        return sum(1 for n in self.graph.nodes 
                   if self.graph.nodes[n]['type'] == node_type and self.graph.nodes[n]['active'])

    def get_total_occupancy(self, node_type):
        total = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] == node_type:
                total += len(self.grid.get_cell_list_contents([node]))
        return total
        
    def get_occupancy(self, node):
        count = len(self.grid.get_cell_list_contents([node]))
        cap = self.graph.nodes[node]['capacity']
        return count / cap if cap > 0 else 1

    def move_agent(self, agent, old_node, new_node):
        self.grid.move_agent(agent, new_node)

    def remove_agent_from_node(self, agent, node):
        self.grid.remove_agent(agent)

    def spawn_agents(self):
        row = self.schedule_data[self.schedule_data['hour'] == self.current_hour]
        if row.empty: return

        arr_count = row.iloc[0]['arr_agents']
        dep_count = row.iloc[0]['dep_agents']

        num_new_dep = int(np.random.poisson(dep_count / 60.0))
        for _ in range(num_new_dep):
            a = AirportAgent(self.next_id(), self, 'departure', 'ME')
            self.schedule.add(a)
            self.grid.place_agent(a, 'ME')

        num_new_arr = int(np.random.poisson(arr_count / 60.0))
        arr_gates = [n for n in self.graph.nodes 
                     if self.graph.nodes[n]['type'] == 'gate_arr' and self.graph.nodes[n]['active']]
        if not arr_gates: return 
        for _ in range(num_new_arr):
            arr_gates.sort(key=lambda n: self.get_occupancy(n))
            start_gate = arr_gates[0]
            a = AirportAgent(self.next_id(), self, 'arrival', start_gate)
            self.schedule.add(a)
            self.grid.place_agent(a, start_gate)

    def step(self):
        minutes_total = self.current_step * MINUTES_PER_STEP
        new_hour = (minutes_total // 60) % 24
        
        if new_hour != self.current_hour or self.current_step == 0:
            self.current_hour = new_hour
            self.trigger_failures() # Wewnątrz woła update_staffing
            
        if self.optimized_mode and self.current_step % 5 == 0:
             self.trigger_failures() # Odśwież staffing i awarie

        self.spawn_agents()
        self.schedule.step()
        self.datacollector.collect(self)
        self.current_step += 1

# --- DEFINICJE KONFIGURACJI ---

STRATEGIES = {
    "baseline": {
        "description": "Stała alokacja (Wszystko otwarte)",
        "config": {'adaptive': False, 'optimized': False}
    },
    "adaptive": {
        "description": "Adaptacja Standard (Zaokrąglanie w górę)",
        "config": {'adaptive': True, 'optimized': False}
    },
    "optimized": {
        "description": "Adaptacja Agresywna (Minimalizacja kosztów + tolerancja kolejki)",
        "config": {'adaptive': True, 'optimized': True}
    }
}

EVENTS = {
    "none": {
        "description": "Brak zakłóceń",
        "config": {}
    },
    "random_failures": {
        "description": "Losowe awarie Security/Gate (20% szans)",
        "config": {'security_prob': 0.2, 'gate_prob': 0.2}
    },
    "security_blackout": {
        "description": "CRITICAL: Security Down 2:00-6:00",
        "config": {'scheduled_downtime': {'security': (2, 4)}}
    }
}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    parser = argparse.ArgumentParser(description="Symulacja Lotniska (Mesa)")
    parser.add_argument("strategy", nargs="?", choices=STRATEGIES.keys(), help="Strategia alokacji zasobów")
    parser.add_argument("event", nargs="?", choices=EVENTS.keys(), help="Scenariusz wydarzeń/awarii")
    parser.add_argument("--list", action="store_true", help="Wyświetl dostępne opcje")
    
    args = parser.parse_args()

    if args.list:
        print("\nDOSTĘPNE STRATEGIE:")
        for k, v in STRATEGIES.items(): print(f"  - {k}: {v['description']}")
        print("\nDOSTĘPNE WYDARZENIA:")
        for k, v in EVENTS.items(): print(f"  - {k}: {v['description']}")
        sys.exit(0)

    # Domyślne wartości jeśli nie podano
    if not args.strategy: args.strategy = "baseline"
    if not args.event: args.event = "none"

    strat_conf = STRATEGIES[args.strategy]
    event_conf = EVENTS[args.event]
    
    # Łączenie konfiguracji
    full_config = {**strat_conf['config'], **event_conf['config']}

    print(f"\n>>> SYMULACJA START")
    print(f"Strategia: {args.strategy.upper()} ({strat_conf['description']})")
    print(f"Wydarzenie: {args.event.upper()} ({event_conf['description']})")
    
    model = AirportModel(schedule_df, config=full_config)
    
    for _ in range(24 * 60):
        model.step()
        
    results = model.datacollector.get_model_vars_dataframe()
    
    # Przetwarzanie danych przepływu
    flow_cols = ["Flow_CheckIn_Cum", "Flow_Security_Cum", "Flow_GateDep_Cum", "Flow_GateArr_Cum", "Flow_Baggage_Cum"]
    df_flow = results[flow_cols].copy()
    df_flow['Hour'] = df_flow.index // 60
    hourly_flow = df_flow.groupby('Hour').max().diff().fillna(df_flow.groupby('Hour').max())
    
    # WIZUALIZACJA
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    time_hours = results.index / 60.0
    
    # 1. Kolejki
    ax1 = axes[0]
    ax1.plot(time_hours, results["CheckIn_Queue"], label="Check-In", color='orange')
    ax1.plot(time_hours, results["Security_Queue"], label="Security", color='red')
    if args.strategy == 'optimized':
        ax1.axhline(y=50, color='gray', linestyle=':', label="Próg tolerancji")
    ax1.set_title(f"Kolejki Pasażerów [{args.strategy} + {args.event}]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Zasoby
    ax2 = axes[1]
    ax2.plot(time_hours, results["Active_CheckIns"], label="Otwarte Check-In", drawstyle='steps-post')
    ax2.plot(time_hours, results["Active_Security"], label="Otwarte Security", drawstyle='steps-post')
    ax2.set_title("Zasoby (Otwarte stanowiska)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Suma kolejek
    ax3 = axes[2]
    ax3.plot(time_hours, results["CheckIn_Queue"] + results["Security_Queue"], label="Total Queue", color='purple')
    ax3.set_title("Całkowite obciążenie systemu")
    ax3.grid(True, alpha=0.3)
    
    # 4. Przepływ
    ax4 = axes[3]
    width = 0.15
    h = hourly_flow.index
    ax4.bar(h - 2*width, hourly_flow["Flow_CheckIn_Cum"], width, label="Check-In Out")
    ax4.bar(h - width, hourly_flow["Flow_Security_Cum"], width, label="Security Out")
    ax4.bar(h, hourly_flow["Flow_GateDep_Cum"], width, label="Departures")
    ax4.bar(h + width, hourly_flow["Flow_GateArr_Cum"], width, label="Arrivals")
    ax4.bar(h + 2*width, hourly_flow["Flow_Baggage_Cum"], width, label="Baggage")
    
    ax4.set_title("Przepustowość godzinowa")
    ax4.set_xlabel("Godzina")
    ax4.set_ylabel("Pax/h")
    ax4.legend(loc="upper right", fontsize='small')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(0, 25))

    plt.tight_layout()
    plt.show()
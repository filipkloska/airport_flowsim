import mesa
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
# Skala 1:1 - jeden agent to jeden prawdziwy pasażer.
# UWAGA: Przy 25 000 pax/h symulacja może działać wolniej.
SCALING_FACTOR = 1 

# Definicja stref
# Check-in: generujemy 20 punktów wzdłuż osi X (np. od x=10 do x=90 co 4 pola)
checkin_points = [(x, 10) for x in range(10, 90, 4)][:20]

ZONES = {
    'entrance': {
        'coords': [(x, 0) for x in range(10, 90)], 
        'threshold': 1000,  # ZWIĘKSZONO
        'name': "Wejście Główne"
    },
    'baggage': {
        'coords': [(x, 15) for x in range(10, 90)], 
        'threshold': 200,   # ZWIĘKSZONO
        'name': "Odbiór Bagażu"
    },
    'check_in': {
        'coords': checkin_points, 
        'threshold': 50 * 20, # 50 osób na jedno stanowisko * 20 stanowisk = 1000 łącznie
        'name': "Strefa Check-in (20 stanowisk)"
    },
    'security': {
        'coords': [(x, 30) for x in range(20, 80, 5)], 
        'threshold': 50,      # BEZ ZMIAN (Wąskie gardło)
        'name': "Kontrola Bezpieczeństwa"
    },
    'gates': {
        'coords': [(x, 48) for x in range(5, 95, 5)], 
        'threshold': 80,      # BEZ ZMIAN
        'name': "Gate'y"
    }
}

class PassengerAgent(mesa.Agent):
    def __init__(self, unique_id, model, flow_type):
        super().__init__(model) 
        self.unique_id = unique_id
        self.flow_type = flow_type 
        self.timer = 0
        
        if self.flow_type == 'departing':
            self.state = "TO_CHECKIN"
            # Wybór losowego stanowiska check-in z listy 20 dostępnych
            pts = ZONES['check_in']['coords']
            self.target = pts[np.random.randint(0, len(pts))]
        else:
            self.state = "TO_BAGGAGE"
            bg_coords = ZONES['baggage']['coords']
            self.target = bg_coords[np.random.randint(0, len(bg_coords))]

    def move_towards(self, target):
        if self.pos is None: return

        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        if not possible_steps: return

        # Wybierz krok najbliższy celowi
        best_step = min(possible_steps, key=lambda p: np.linalg.norm(np.array(p) - np.array(target)))
        
        # Ruch bez blokowania (tłum przenika)
        self.model.grid.move_agent(self, best_step)

    def step(self):
        if self.pos is None:
            self.model.schedule.remove(self)
            return

        # --- ODLATUJĄCY ---
        if self.flow_type == 'departing':
            if self.state == "TO_CHECKIN":
                # Tolerancja zbliżenia (żeby nie musieli wchodzić idealnie w punkt)
                if np.linalg.norm(np.array(self.pos) - np.array(self.target)) < 2:
                    self.state = "TO_SECURITY"
                    sec_coords = ZONES['security']['coords']
                    # Idź do najbliższego punktu kontroli
                    self.target = min(sec_coords, key=lambda t: np.linalg.norm(np.array(self.pos)-np.array(t)))
                else:
                    self.move_towards(self.target)

            elif self.state == "TO_SECURITY":
                if np.linalg.norm(np.array(self.pos) - np.array(self.target)) < 2:
                    self.state = "IN_SECURITY"
                    self.timer = np.random.randint(2, 5) 
                else:
                    self.move_towards(self.target)

            elif self.state == "IN_SECURITY":
                if self.timer > 0:
                    self.timer -= 1
                else:
                    self.state = "TO_GATE"
                    x, y = self.pos
                    if y+2 < self.model.grid.height:
                        self.model.grid.move_agent(self, (x, y+2))
                    gate_coords = ZONES['gates']['coords']
                    self.target = gate_coords[np.random.randint(0, len(gate_coords))]

            elif self.state == "TO_GATE":
                if np.linalg.norm(np.array(self.pos) - np.array(self.target)) < 2:
                    self.model.grid.remove_agent(self)
                    self.model.schedule.remove(self)
                else:
                    self.move_towards(self.target)

        # --- PRZYLATUJĄCY ---
        elif self.flow_type == 'arriving':
            if self.state == "TO_BAGGAGE":
                if np.linalg.norm(np.array(self.pos) - np.array(self.target)) < 2:
                    self.state = "COLLECTING_BAGS"
                    self.timer = np.random.randint(5, 10) 
                else:
                    self.move_towards(self.target)
            
            elif self.state == "COLLECTING_BAGS":
                if self.timer > 0:
                    self.timer -= 1
                else:
                    self.state = "TO_EXIT"
                    self.target = (self.pos[0], 0) 

            elif self.state == "TO_EXIT":
                if self.pos[1] == 0: 
                    self.model.grid.remove_agent(self)
                    self.model.schedule.remove(self)
                else:
                    self.move_towards(self.target)

class AirportModel(mesa.Model):
    def __init__(self, csv_path="dane.csv"):
        super().__init__()
        self.width = 100
        self.height = 50
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)
        self.current_id = 0
        self.hourly_data = self.read_csv_dual(csv_path)
        
    def read_csv_dual(self, filepath):
        print(f"📂 Wczytuję dane: {filepath}")
        data_map = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                if "Time" in line or not line.strip(): continue
                parts = line.strip().split()
                if len(parts) < 3: continue
                try:
                    hour = int(parts[0].split(':')[0])
                    def parse_val(val_str):
                        v = float(val_str.replace(',', '.'))
                        if v < 1000: return int(v * 1000)
                        return int(v)
                    arr_pax = parse_val(parts[1]) 
                    dep_pax = parse_val(parts[2]) 
                    data_map[hour] = {'arr': arr_pax, 'dep': dep_pax}
                except ValueError: continue
            return data_map
        except Exception:
            return {}

    def spawn_agents(self):
        current_step = self.schedule.steps
        current_hour = (current_step // 60) % 24
        
        if current_hour in self.hourly_data:
            data = self.hourly_data[current_hour]
            
            # Odloty
            rate_dep = (data['dep'] / 60) * SCALING_FACTOR
            for _ in range(np.random.poisson(rate_dep)):
                a = PassengerAgent(self.current_id, self, 'departing')
                self.current_id += 1
                self.schedule.add(a)
                # Losowe wejście na dole
                self.grid.place_agent(a, (np.random.randint(10, 90), 0))

            # Przyloty
            rate_arr = (data['arr'] / 60) * SCALING_FACTOR
            for _ in range(np.random.poisson(rate_arr)):
                gate_coords = ZONES['gates']['coords']
                start_gate = gate_coords[np.random.randint(0, len(gate_coords))]
                a = PassengerAgent(self.current_id, self, 'arriving')
                self.current_id += 1
                self.schedule.add(a)
                # Stawiamy agenta nawet na zajętym polu (Multigrid)
                self.grid.place_agent(a, start_gate)

    def analyze_bottlenecks(self):
        alerts = []
        
        # 1. Analiza ogólna stref
        current_counts = {k: 0 for k in ZONES.keys()}
        
        # Przeliczamy agentów (szybsza metoda niż iteracja po strefach w pętli)
        # Agent sam nie wie w jakiej jest strefie, więc sprawdzamy pozycję
        for agent in self.schedule.agents:
            if agent.pos is None: continue
            
            # Sprawdzamy czy pozycja agenta należy do którejś strefy
            # Optymalizacja: można by to cache'ować, ale przy 100x50 mapie jest ok.
            for zone_key, zone_data in ZONES.items():
                if agent.pos in zone_data['coords']:
                    current_counts[zone_key] += 1
                    break # Agent jest tylko w jednej strefie na raz

        # 2. Generowanie alertów
        for zone_key, count in current_counts.items():
            threshold = ZONES[zone_key]['threshold']
            name = ZONES[zone_key]['name']
            
            if count > threshold:
                alerts.append(f"🔴 {name}: {count} os. (Limit: {threshold})")
            elif count > threshold * 0.8: # Ostrzeżenie przy 80% wypełnienia
                alerts.append(f"⚠️ {name}: {count} os.")
                
        return alerts

    def step(self):
        self.spawn_agents()
        self.schedule.step()

if __name__ == "__main__":
    model = AirportModel("dane.csv")
    print("🚀 Analiza Wąskich Gardeł (Skala 1:1)...")
    print("-" * 60)
    
    # Symulacja 24h (1440 minut)
    for i in range(1440): 
        model.step()
        
        # Raport co godzinę
        if i % 60 == 0:
            h = (i // 60) % 24
            bottlenecks = model.analyze_bottlenecks()
            total = model.schedule.get_agent_count()
            
            print(f"\n🕓 GODZINA {h:02d}:00 | Agentów: {total}")
            
            if not bottlenecks:
                print("   🟢 Przepływ stabilny.")
            else:
                print("   💀 ZATORY:")
                for alert in bottlenecks:
                    print(f"      {alert}")
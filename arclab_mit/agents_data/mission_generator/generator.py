import random


class OrbitPool:
    def __init__(self):
        # Define the min and max values for each orbital parameter
        self.ranges = {
            'SMA': (724999.9, 754500.1875),
            'ECC': (0, 0.034483000636100769),
            'INC': (0, 1),
            'LPE': (0, 355),
            'LAN': (0, 350),
            'MNA': (5.93062879827673231, 6.1959188445798699),
            'EPH': (0, 0),  # Adjusted to a realistic max for example purposes
            'REF': (1, 1)
        }

    def generate_random_orbit(self):
        # Generate random values for each orbital parameter within its range
        orbit_values = tuple(random.uniform(min_val, max_val) for param, (min_val, max_val) in self.ranges.items())
        return orbit_values

class Generator:
    def __init__(self):
        self.kerbal_path = r'C:\Kerbal Space Program\saves\missions\pe1_i3\pe1_i3_init.sfs'
        self.color_info_1 = '\033[92m'
        self.color_info_2 = '\033[94m'
        self.color_info_3 = '\033[95m'
        self.color_end = '\033[0m'
        self.orbit_pool = OrbitPool()

    def generate_orbit(self, sma, ecc, inc, lpe, lan, mna, eph, ref):
        # Generating orbit parameters based on inputs
        orbit_params = f'''                 
            ORBIT
            {{
                SMA = {sma}
                ECC = {ecc}
                INC = {inc}
                LPE = {lpe}
                LAN = {lan}
                MNA = {mna}
                EPH = {int(eph)}
                REF = {int(ref)}
            }}
        '''
        return orbit_params.strip()

    def modify_evader_orbit(self):
        # Generate random orbit parameters for the evader
        orbit_params = self.orbit_pool.generate_random_orbit()
        return self.generate_orbit(*orbit_params)

    def modify_pursuer_orbit(self):
        # Generate random orbit parameters for the pursuer
        orbit_params = self.orbit_pool.generate_random_orbit()
        return self.generate_orbit(*orbit_params)

    def parse_and_rewrite_mission_file(self):
        with open(self.kerbal_path, 'r') as file:
            lines = file.readlines()

        processed_lines = []
        inside_vessel = False
        is_pursuer = False
        is_evader = False
        brace_count = 0

        for line in lines:
            line = line.strip()

            if 'VESSEL' in line:
                print(self.color_info_1 + "Inside Vessel" + self.color_end)
                inside_vessel = True
                brace_count = 0
                processed_lines.append(line)  # Append the line here
                continue

            if inside_vessel:
                if '{' in line:
                    brace_count += 1
                if '}' in line:
                    brace_count -= 1

            if brace_count == 0 and inside_vessel:
                print(self.color_info_2 + "Outside Vessel" + self.color_end)
                inside_vessel = False
                is_pursuer = False
                is_evader = False

            if inside_vessel and not is_evader and not is_pursuer:
                if 'name = Evader' in line:
                    is_evader = True
                    is_pursuer = False
                elif 'name = Pursuer' in line:
                    is_pursuer = True
                    is_evader = False

            if inside_vessel and is_evader and 'ORBIT' in line:
                modified_line = self.modify_evader_orbit()
                print(self.color_info_3 + modified_line + self.color_end)
                processed_lines.append(modified_line)
                continue

            if inside_vessel and is_pursuer and 'ORBIT' in line:
                modified_line = self.modify_pursuer_orbit()
                print(self.color_info_3 + modified_line + self.color_end)
                processed_lines.append(modified_line)
                continue

            processed_lines.append(line)

        # Now write the processed lines back to the file
        with open(self.kerbal_path, 'w') as file:
            for line in processed_lines:
                file.write(line + '\n')
            file.close()

if __name__ == '__main__':
    gen = Generator()
    gen.parse_and_rewrite_mission_file()

    print("Mission file updated.")

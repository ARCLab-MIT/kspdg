import random
from orbit import sample_n_orbits


class OrbitPool:
    def __init__(self):
        # Define the min and max values for each orbital parameter
        self.orbits = []
        self.ranges = {
#            'n_orbits': 10,
            'n_orbits': 10,
            'dmin': [500, 1000],
            'dmax': [5000, 10000],
            'speed_percent_range': [0, 100]
        }

    def retrieve_random_orbit(self):
        # Generate random values for each orbital parameter within its range
        if len(self.orbits) == 0:
            self.orbits = self.generate_orbit_pool()
        return self.orbits.pop()

    def generate_orbit_pool(self):
        # Generate a pool of orbits using the provided sample_n_orbits function
        n_orbits = self.ranges['n_orbits']
        dmin = random.randint(*self.ranges['dmin'])
        dmax = random.randint(*self.ranges['dmax'])
        speed_percent_range = random.randint(*self.ranges['speed_percent_range'])

        return sample_n_orbits(n_orbits, dmin, dmax, speed_percent_range)

    def get_random_range_value(self, range_list):
        # Helper function to get a random value from a range list
        return random.randint(*range_list)


class Generator:
    def __init__(self):
        self.kerbal_path = r'C:\Kerbal Space Program\saves\missions\pe1_i3\pe1_i3_init.sfs'
        self.color_info_1 = '\033[92m'
        self.color_info_2 = '\033[94m'
        self.color_info_3 = '\033[95m'
        self.color_end = '\033[0m'
        self.orbit_pool = OrbitPool()

    def generate_orbit(self, sma, ecc, inc, lpe, lan, mna, eph=0, ref=1):
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
        orbit_params = [
            750000,
            0,
            0.0001,
            0,
            0,
            5.9341194567807207
        ]
        return self.generate_orbit(*orbit_params)

    def modify_pursuer_orbit(self):
        # Retrieve the orbit instance, assuming it's stored or created somewhere in the class
        my_orbit_instance = self.orbit_pool.retrieve_random_orbit()  # This method should return an instance of MyOrbit
        orbit_params = my_orbit_instance.get_keplerian_elements()

        # Pass all the keplerian elements
        return self.generate_orbit(*orbit_params)

    def parse_and_rewrite_mission_file(self):
        with open(self.kerbal_path, 'r') as file:
            lines = file.readlines()

        processed_lines = []
        inside_vessel = False
        inside_orbit = False
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
                    if inside_orbit:
                        inside_orbit = False
                        continue

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

            if inside_orbit:
                continue

            if inside_vessel and is_evader and 'ORBIT' == line:
                inside_orbit = True
                modified_line = self.modify_evader_orbit()
                print("EVADER")
                print(self.color_info_3 + modified_line + self.color_end)
                processed_lines.append(modified_line)
                continue

            if inside_vessel and is_pursuer and 'ORBIT' == line:
                inside_orbit = True
                modified_line = self.modify_pursuer_orbit()
                print("PURSUER")
                print(self.color_info_3 + modified_line + self.color_end)
                processed_lines.append(modified_line)
                continue

            processed_lines.append(line)

        # Now write the processed lines back to the file
        with open(self.kerbal_path, 'w') as file:
            for line in processed_lines:
                file.write(line + '\n')
            file.close()


def main():
    gen = Generator()
    gen.parse_and_rewrite_mission_file()

    print("Mission file updated.")


if __name__ == '__main__':
    main()

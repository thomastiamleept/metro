# This is a dictionary containing the distance information between each pair of stations.
# The distances in the azul, amerale, and vermelha lines are taken from the information
# in Wikipedia, whereas the distances in the verde line are estimated from Google maps
# as there is no available information.
stations_distance = {'AP' : {'EN' : 1.4 },
                     'AM' : {'SA' : 1.3, 'SA' : 1.3, 'OL' : 0.8, 'AE' : 0.7, 'AN' : 1.2},
                     'AF' : {'PO' : 0.9, 'AS' : 1.2 },
                     'AH' : {'LA' : 0.6, 'CM' : 0.8 },
                     'AL' : {'CG' : 1.5, 'RM' : 0.5 },
                     'AS' : {'AF' : 1.2, 'RB' : 0.9 },
                     'AX' : {'SR' : 1.7, 'LU' : 0.9 },
                     'AN' : {'AM' : 1.2, 'AM' : 1.2, 'IN' : 0.5 },
                     'AE' : {'RM' : 1.0, 'AM' : 0.7, 'AM' : 0.7 },
                     'AV' : {'RE' : 0.5, 'MP' : 0.6, 'MP' : 0.6 },
                     'BC' : {'TP' : 0.9, 'RE' : 0.7, 'RO' : 0.8, 'CS' : 0.7 },
                     'BV' : {'OL' : 1.3, 'CH' : 0.8 },
                     'CR' : {'OS' : 0.6, 'OR' : 0.8 },
                     'CS' : {'BC' : 0.7 },
                     'CG' : {'QC' : 0.9, 'CU' : 0.8, 'TE' : 0.7, 'AL' : 1.5 },
                     'CP' : {'EC' : 0.7, 'SA' : 0.7, 'SA' : 0.7 },
                     'CA' : {'CM' : 0.9, 'PO' : 0.7 },
                     'CH' : {'BV' : 0.8, 'OS' : 0.7},
                     'CU' : {'CG' : 0.8, 'EC' : 1.1 },
                     'CM' : {'AH' : 0.8, 'CA' : 0.9 },
                     'EN' : {'MO' : 0.8, 'AP' : 1.4 },
                     'EC' : {'CU' : 1.1, 'CP' : 0.7 },
                     'IN' : {'AN' : 0.5, 'MM' : 0.7 },
                     'JZ' : {'PE' : 0.8, 'LA' : 1.1 },
                     'LA' : {'JZ' : 1.1, 'AH' : 0.6 },
                     'LU' : {'AX' : 0.9, 'QC' : 1.0 },
                     'MP' : {'AV' : 0.6, 'PA' : 0.6, 'PI' : 0.7, 'RA' : 0.7 },
                     'MM' : {'IN' : 0.7, 'RO' : 0.3 },
                     'MO' : {'OR' : 1.0, 'EN' : 0.8 },
                     'OD' : {'SR' : 1.0 },
                     'OS' : {'CH' : 0.7, 'CR' : 0.6 },
                     'OL' : {'AM' : 0.8, 'AM' : 0.8, 'BV': 1.3 },
                     'OR' : {'CR' : 0.8, 'MO' : 1.0 },
                     'PA' : {'MP' : 0.6, 'MP' : 0.6, 'SS' : 0.7, 'SS' : 0.7 },
                     'PI' : {'SA' : 0.7, 'SA' : 0.7, 'MP' : 0.7, 'MP' : 0.7 },
                     'PO' : {'CA' : 0.7, 'AF' : 0.9 },
                     'PE' : {'SS' : 0.7, 'SS' : 0.7, 'JZ' : 0.8 },
                     'QC' : {'LU' : 1.0, 'CG' : 0.9 },
                     'RA' : {'MP' : 0.7, 'MP' : 0.7 },
                     'RB' : {'AS' : 0.9 },
                     'RE' : {'BC' : 0.7, 'AV' : 0.5 },
                     'RM' : {'AL' : 0.5, 'AE' : 1.0 },
                     'RO' : {'MM' : 0.3, 'BC' : 0.8 },
                     'SA' : {'CP' : 0.7, 'PI' : 0.7, 'SS' : 0.6, 'SS' : 0.6, 'AM' : 1.3, 'AM' : 1.3 },
                     'SA' : {'CP' : 0.7, 'PI' : 0.7, 'SS' : 0.6, 'SS' : 0.6, 'AM' : 1.3, 'AM' : 1.3 },
                     'SP' : {'TP' : 1.1 },
                     'SS' : {'PA' : 0.7, 'PE' : 0.7, 'SA' : 0.6, 'SA' : 0.6 },
                     'SS' : {'PA' : 0.7, 'PE' : 0.7, 'SA' : 0.6, 'SA' : 0.6 },
                     'SR' : {'OD' : 1.0, 'AX' : 1.7 },
                     'TE' : {'CG' : 0.7 },
                     'TP' : {'SP' : 1.1, 'BC' : 0.9 }}

# This dictionary contains all the stations with secondary exits, and the corresponding
# code for the other exit associated with them.
stations_alternate = {'AM1': 'AM2',
                      'MP1': 'MP2',
                      'SA1': 'SA2',
                      'SS1': 'SS2'}

# These arrays contain the stations in each of the four lines in the Lisbon metro.
# The list contains the station codes in the line in sequential order. If a station
# is associated with multiple codes, only the primary code is stated in this list.
line_azul = ['SP', 'TP', 'BC', 'RE', 'AV', 'MP1', 'PA', 'SS1', 'PE', 'JZ', 'LA', 'AH', 'CM', 'CA', 'PO', 'AF', 'AS', 'RB']
line_amarela = ['OD', 'SR', 'AX', 'LU', 'QC', 'CG', 'CU', 'EC', 'CP', 'SA1', 'PI', 'MP1', 'RA']
line_verde = ['TE', 'CG', 'AL', 'RM', 'AE', 'AM1', 'AN', 'IN', 'MM', 'RO', 'BC', 'CS']
line_vermelha = ['SS1', 'SA1', 'AM1', 'OL', 'BV', 'CH', 'OS', 'CR', 'OR', 'MO', 'EN', 'AP']

# This function gets a list of stations from a given line.
# The type parameter can either be:
# 'p' (partial): gets the stations as is (for stations with multiple exits,
#                only the first exit appears) [default]
# 'b' (base):    gets the base stations (stations with multiple exits are treated as one)
# 'f' (full):    gets the full stations (all exits appear)
def s_get_stations(line, type='p', reverse=False):
    result = []
    for station in line:
        if type == 'b':
            result.append(station[:2])
        elif type == 'f' and station in stations_alternate:
            result.append(station)
            result.append(stations_alternate[station])
        else:
            result.append(station)
    if reverse == True:
        result.reverse()
    return result

# This function returns the line color of the two given station codes s1 and s2.
# If the two stations given are not directly connected, None is returned.
# The function returns a tuple (a, b), where:
# a: the name of the line which is either 'azul', 'amarela', 'verde', 'vermelha'
# b: True if the direction is forward and False if the direction is backward
def s_get_line(s1, s2):
    line_names = ['azul', 'amarela', 'verde', 'vermelha']
    lines = [s_get_stations(line_azul, 'b', reverse=False),
             s_get_stations(line_amarela, 'b', reverse=False),
             s_get_stations(line_verde, 'b', reverse=False),
             s_get_stations(line_vermelha, 'b', reverse=False)]
    for i in range(len(lines)):
        current_line = lines[i]
        for j in range(len(current_line)):
            if j < len(current_line) - 1 and current_line[j] == s1 and current_line[j + 1] == s2:
                return (line_names[i], False)
            elif j > 0 and current_line[j] == s1 and current_line[j - 1] == s2:
                return (line_names[i], True)
    return None

# This function returns the direct distance between two station codes s1 and s2.
# If the two stations given are not directly connected,-1 is returned.
def s_dist(s1, s2):
    if not s1 in stations_distance or not s2 in stations_distance:
        return -1
    if not s2 in stations_distance[s1]:
        return -1
    return stations_distance[s1][s2]

# This function returns the total distance of a given line. The line is a list
# following the format of the list representing the four lines in the Lisbon
# metro as defined above. The line should be in base form.
def s_total_dist(base_line):
    total_distance = 0
    for index, station in enumerate(base_line):
        if index == 0:
            continue
        previous_station = base_line[index - 1]
        total_distance += stations_distance[previous_station][station]
    return total_distance

# This function returns all possible paths from s1 to s2. s1 and s2 should
# be both in base form. This functions uses depth first search algorithm (DFS)
# to extract the paths.
def s_get_paths(s1, s2):
    if not s1 in stations_distance or not s2 in stations_distance:
        return []
    paths = []
    stack = []
    stack.append({'path': [s1], 'distance': 0.0})
    while len(stack) != 0:
        top = stack.pop()
        current = top['path'][-1]
        if current == s2:
            paths.append(top)
            continue
        connections = stations_distance[current]
        for connection, distance in connections.items():
            if connection in stations_alternate:
                continue
            if connection in top['path']:
                continue
            new_path = []
            new_path.extend(top['path'])
            new_path.append(connection)
            new_distance = top['distance'] + distance
            stack.append({'path': new_path, 'distance': new_distance})
    return sorted(paths, key = lambda i: i['distance'])

# This function takes a list of paths and formats it according to the 
# exit time estimation algorithm.
def s_format_paths(paths):
    new_paths = []
    path_details = []
    for p in paths:
        path = p['path']
        lines = []
        for i in range(1, len(path)):
            src = path[i - 1]
            dest = path[i]
            line = s_get_line(src, dest)
            lines.append(line)
        current_line = None
        starting_station = path[0]
        recent_station = path[0]
        res = []
        transfers = 0
        for i in range(1, len(path)):
            c = lines[i - 1]
            if c != current_line:
                if current_line is not None:
                    transfers = transfers + 1
                    res.append((current_line[0],
                                current_line[1],
                                starting_station,
                                recent_station))
                current_line = c
                starting_station = recent_station
            recent_station = path[i]
        if current_line is not None:
            res.append((current_line[0],
                        current_line[1],
                        starting_station,
                        recent_station))
        details = {}
        details['rail_distance'] = p['distance']
        details['transfers'] = transfers
        details['stations'] = len(path) - 1
        new_paths.append(res)
        path_details.append(details)
    return new_paths, path_details

def s_get_station_list_from_path(path):
    lines = [s_get_stations(line_azul, 'b', reverse=False),
             s_get_stations(line_amarela, 'b', reverse=False),
             s_get_stations(line_verde, 'b', reverse=False),
             s_get_stations(line_vermelha, 'b', reverse=False)]
    res = []
    for p in path:
        src = p[2]
        dest = p[3]
        target = None
        for line in lines:
            if src in line and dest in line:
                target = line
        if not p[1]:
            for i in range(target.index(src), target.index(dest) + 1):
                if len(res) == 0 or res[len(res) - 1] != target[i]:
                    res.append(target[i])
        else:
            for i in range(target.index(src), target.index(dest) - 1, -1):
                if len(res) == 0 or res[len(res) - 1] != target[i]:
                    res.append(target[i])
    return res
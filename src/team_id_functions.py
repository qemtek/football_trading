# Get team ID from name
def fetch_id(team_name, cursor):
    cursor.execute("Select team_id from team_ids where team_name == ?", [team_name])
    return cursor.fetchone()[0]


# Get name from team ID
def fetch_name(team_id, cursor):
    cursor.execute("Select team_name from team_ids where team_id == ?", [team_id])
    return cursor.fetchone()[0]


# Change FPL team ID to the ID we use.
def fix_id(id):
    if id == 1:  # Arsenal
        return 1
    if id == 2:  # Bournemouth
        return 2
    if id == 3:  # Brighton
        return 21
    if id == 4:  # Burnley
        return 3
    if id == 5:  # Cardiff
        return 24
    if id == 6:  # Chelsea
        return 4
    if id == 7:  # Crystal Palace
        return 5
    if id == 8:  # Everton
        return 6
    if id == 9:  # Fulham
        return 25
    if id == 10:  # Huddersfield
        return 22
    if id == 11:  # Leicester
        return 8
    if id == 12:  # Liverpool
        return 9
    if id == 13:  # Man City
        return 10
    if id == 14:  # Man United
        return 11
    if id == 15:  # Newcastle
        return 23
    if id == 16:  # Southampton
        return 13
    if id == 17:  # Tottenham
        return 17
    if id == 18:  # Watford
        return 18
    if id == 19:  # West Ham
        return 20
    if id == 20:  # Wolves
        return 26


import Levenshtein

from football_trading.src.utils.db import run_query, connect_to_db


def fetch_id(team_name):
    """Get team ID from name"""
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(
        "Select team_id from team_ids where team_name = '{}' or alternate_name = '{}'".format(
        team_name, team_name))
    output = cursor.fetchone()[0]
    conn.close()
    return output


def fetch_name(team_id):
    """Get name from team ID"""
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("Select team_name from team_ids where team_id == ?", [team_id])
    output = cursor.fetchone()[0]
    conn.close()
    return output


def find_closest_match(team_name):
    # Hardcoded some of the matches because it was too difficult to match accurately
    if team_name == 'Man City':
        output = {"team_name": 'Manchester City', "team_id": fetch_id("Manchester City")}
        return output
    elif team_name == 'Spurs':
        output = {"team_name": 'Tottenham', "team_id": fetch_id("Tottenham")}
        return output
    else:
        conn = connect_to_db()
        df = run_query("Select team_name, team_id from team_ids")
        df['l_dist'] = df['team_name'].apply(lambda x: Levenshtein.ratio(x, team_name))
        max_similarity = max(df['l_dist'])
        closest_match = df.loc[df['l_dist'] == max_similarity,
                               ['team_name', 'team_id']].reset_index(drop=True)
        output = {"team_name": closest_match.loc[0, 'team_name'],
                  "team_id": closest_match.loc[0, 'team_id']}
        conn.close()
        return output


def fetch_alternative_name(team_name):
    """If a team name cannot be found, use this to try an alternative name"""
    if team_name == 'Birmingham City':
        return 'Birmingham'
    elif team_name == 'Blackburn Rovers':
        return 'Blackburn'
    elif team_name == 'Bolton Wanderers':
        return 'Bolton'
    elif team_name == 'Bradford City':
        return 'Bradford'
    elif team_name == 'Brighton & Hove Albion':
        return 'Brighton'
    elif team_name == 'Cardiff City':
        return 'Cardiff'
    elif team_name == 'Charlton Athletic':
        return 'Charlton'
    elif team_name == 'Coventry City':
        return 'Coventry'
    elif team_name == 'Derby County':
        return 'Derby'
    elif team_name == 'Huddersfield Town':
        return 'Huddersfield'
    elif team_name == 'Hull City':
        return 'Hull'
    elif team_name == 'Ipswich Town':
        return 'Ipswich'
    elif team_name == 'Leeds United':
        return 'Leeds'
    elif team_name == 'Leicester City':
        return 'Leicester'
    elif team_name == 'Manchester City':
        return 'Man City'
    elif team_name == 'Manchester United':
        return 'Man United'
    elif team_name == 'Newcastle United':
        return 'Newcastle'
    elif team_name == 'Nottingham Forest':
        return 'Nottingham'
    elif team_name == 'Norwich City':
        return 'Norwich'
    elif team_name == 'Oldham Athletic':
        return 'Oldham'
    elif team_name == 'Portsmouth':
        return 'Portsmouth'
    elif team_name == 'Queens Park Rangers':
        return 'QPR'
    elif team_name == 'Stoke City':
        return 'Stoke'
    elif team_name == 'Swansea City':
        return 'Swansea'
    elif team_name == 'Swindon Town':
        return 'Swindon'
    elif team_name == 'Tottenham Hotspur':
        return 'Tottenham'
    elif team_name == 'West Bromwich Albion':
        return 'West Brom'
    elif team_name == 'West Ham United':
        return 'West Ham'
    elif team_name == 'Wigan Athletic':
        return 'Wigan'
    elif team_name == 'Wolverhampton Wanderers':
        return 'Wolves'
    else:
        return ''


def fetch_alternative_name2(team_name):
    """If a team name cannot be found, use this to try an alternative name"""
    if team_name == 'Manchester United':
        return 'Man Utd'
    elif team_name == 'Tottenham Hotspur':
        return 'Spurs'
    elif team_name == 'Sheffield United':
        return 'Sheff Utd'
    elif team_name == 'Wolverhampton Wonderers':
        return 'Wolverhampton'
    else:
        return ''

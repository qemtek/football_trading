def fetch_id(team_name, cursor):
    """Get team ID from name"""
    cursor.execute(
        "Select team_id from team_ids where team_name == ? or alternate_name == ?",
        [team_name, team_name])
    return cursor.fetchone()[0]


def fetch_name(team_id, cursor):
    """Get name from team ID"""
    cursor.execute("Select team_name from team_ids where team_id == ?", [team_id])
    return cursor.fetchone()[0]


def get_alternative_name(team_name):
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

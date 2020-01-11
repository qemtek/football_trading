import logging
from src.utils.db import connect_to_db, run_query

logger = logging.getLogger('XGBoostModel')

class Model:
    """Anything that can be used by all models goes in this class"""
    def __init__(self):
        self.training_data_query = \
            """select t1.*, m_h.manager home_manager, m_h.start home_manager_start, 
            m_a.manager away_manager, m_a.start away_manager_start 
            from main_fixtures t1 
            left join managers m_h 
            on t1.home_id = m_h.team_id 
            and (t1.date between m_h.start and date(m_h.end, '+1 day') 
            or t1.date > m_h.start and m_h.end is NULL) 
            left join managers m_a 
            on t1.away_id = m_a.team_id 
            and (t1.date between m_a.start and date(m_a.end, '+1 day') 
            or t1.date > m_a.start and m_a.end is NULL) 
            where t1.date > '2013-08-01'"""

    def preprocess(self, X):
        pass

    def train_model(self, X, y):
        pass

    def predict_proba(self, X):
        pass

    def predict_classes(self, X):
        pass

    def get_training_data(self):
        conn, cursor = connect_to_db()
        # Get all fixtures after game week 8, excluding the last game week
        df = run_query(cursor, self.training_data_query)
        return df

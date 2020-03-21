select
    t1.*, m_h.manager home_manager, m_h.start_date home_manager_start,
    m_a.manager away_manager, m_a.start_date away_manager_start
from main_fixtures t1
left join managers m_h
on t1.home_id = m_h.team_id
and (t1.date between m_h.start_date and date(m_h.end_date, '+1 day')
or t1.date > m_h.start_date and m_h.end_date is NULL)
left join managers m_a
on t1.away_id = m_a.team_id
and (t1.date between m_a.start_date and date(m_a.end_date, '+1 day')
or t1.date > m_a.start_date and m_a.end_date is NULL)
where t1.date > '2013-08-01
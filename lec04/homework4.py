def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    all_dates = sorted(birthdays.keys())
    
    for bday in all_dates:
        if bday > date:
            return bday, birthdays[bday]
    
    return all_dates[0], birthdays[all_dates[0]]
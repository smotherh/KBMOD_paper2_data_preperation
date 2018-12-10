from astropy import time

def largest_dt(group):
    """Takes a group [pandas dataframe] and returns a list of tuples containing
    paired elements of the group. Each tuple is a (science, template, dt) triplet
    of pointing visit ids and time stamp difference, matched such that their
    time-stamp difference is the largest among all the pointings in the group.

    Except for groups of length 1, pairing should never fail.
    """
    # pandas dataframes are not awkward to use at all...
    pairs = []
    for (idx, science) in group.iterrows():
        sci_obsdate = science["date_obs"]
        sci_id = science["visit_id"]
        sci_ra = science["ra"]
        sci_dec = science["dec"]
        sci_filename = science["filename"]
        sci_date = time.Time(sci_obsdate.decode("utf-8"))

        maxdt = 0
        for (idx, template) in group.iterrows():
            tmplt_obsdate = template["date_obs"]
            tmplt_id = template["visit_id"]
            tmplt_ra = template["ra"]
            tmplt_dec = template["dec"]
            tmplt_filename = template["filename"]
            tmplt_date = time.Time(tmplt_obsdate.decode("utf-8"))

            # dt is a astropy.time.TimeDelta object and can not be compared to non TimeDelta objects
            # it recognizes positive and negative time delta so we check for its absolute value
            dt = sci_date - tmplt_date
            if abs(dt) > time.TimeDelta(maxdt, format="sec"):
                maxdt = dt
                if sci_id != tmplt_id:
                    pair = (sci_id, tmplt_id, abs(dt))
        try:
            pairs.append(pair)
        except:
            print("Pair not found. Check the group.")

        maxdt = time.TimeDelta(0, format="sec")
    return pairs

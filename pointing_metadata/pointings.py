import pickle

import numpy as np

import astropy.table as astropyTable
from astropy.io.votable import parse


__all__ = ["create_ingest_foldername", "create_imdiff_foldername",
           "read_pointings", "group_pointings", "PointingGroups"]


def create_ingest_foldername(ra, dec):
    """Defines the standard format of folders where data was ingested.

    Parameters
    ----------
    ra : float
        right ascension
    dec : float
        declination
    """
    namestr = "r{0}d{1}"
    return namestr.format(ra, dec)


def create_imdiff_foldername(ra, dec):
    """Defines the standard format of folders where image difference
    data is kept.

    Parameters
    ----------
    ra : float
        right ascension
    dec : float
        declination
    """
    namestr = "imdiff_r{0}d{1}"
    return namestr.format(ra, dec)





def read_pointings(filePath="",maxNight=32, keepOnlyImageType=True):
    """Will read VOT metadata tables for series of VOT table filenames given
    in night_{index}.vot format, where index is incremented index from 0 to
    maxNight and stack them into a singular table. 

    Columns 'archive_file' and 'prodtype' renamed to 'filename' and 'product'.
    Approx. 15 columns are removed as they are not very useful to us.
    Columns 'visit_id', 'reprocessing', 'ingest_folder' and  'imdiff_folder' are
    added. Visit id and reprocessing are essentially just more manageable form
    of existing columns while ingest and imdiff are set to None by default and
    will not meaningfully represent any information untill the pointings have
    been grouped.

    Parameters
    ----------
    maxNight : int
        index marking the last file in series of night_N.vot files that will be
        read.
    keepOnlyImageType : bool
        if True only the products of type "image" will be kept. True by default.
    """
    allNights = []
    for i in range(1, maxNight):
        filename = filePath+"night_{0}.vot".format(i)
        votableFile = parse(filename)
        # also the only table availible
        votable = votableFile.get_first_table() 

        table = votable.to_table()
        table.remove_columns(["md5sum", "filesize", "depth", "seeing", "dtpi",
                              "instrument", "telescope", "surveyid", "dtpropid",
                              "reference", "filter", "proctype", "obsmode",
                              "obstype"])
        col = astropyTable.Column( [i]*len(table) , name="survey_night")
        table.add_column(col)
        allNights.append(table)

    allNights = astropyTable.vstack(allNights)

    allNights.rename_column("archive_file", "filename")
    allNights.rename_column("prodtype", "product")

    if keepOnlyImageType:
        allNights = allNights[allNights["product"] == b"image"]

    visit_ids = map(lambda x: x[-14:-8], allNights["dtacqnam"])
    col = astropyTable.Column( list(visit_ids) , name="visit_id")
    allNights.add_column(col)

    # reprocessing version is hidden in the last section of the full filename
    # so we cut it out into a standalone column to make it easier to work with
    reprocessing_id = map(lambda x:
                          int(x.split(b".")[0][-1:]), allNights["filename"])
    col = astropyTable.Column( list(reprocessing_id), name="reprocessing")
    allNights.add_column(col)

    col = astropyTable.Column( [None]*len(allNights) , name="ingest_folder")
    allNights.add_column(col)

    col = astropyTable.Column( [None]*len(allNights) , name="imdiff_folder")
    allNights.add_column(col)

    # we sort the nights provisionaly just to match Hayden's order, otherwise
    # not required
    allNights.sort(["ra", "dec"])

    return allNights


def group_pointings(pointings, raTol=5e-2, deTol=5e-2, minNumVisits=3,
                    removeSmallGroups=True, keepMaxReprocessingOnly=True):
    """Takes stacked and modified table of pointings, as returned by
    'read_pointings' function and finds groups of pointings based on coordinates.
    The elements of the ingoing table could be modified by this function.

    Same pointing can have several different data reprocessing versions. To
    remove old and keep only the last reprocessing version use keepMaxReprocessingOnly.

    A pointing group is defined as a set of pointings within the neighbourhood
    of (ra, dec) coordinate. The size of that neighbourhood is defined by raTol
    and deTol parameters.

    As primary purpose of this grouping is to create pairs for image differencing
    purposes we can remove trivial groupings containing just a single or a pair
    of pointings. Control this behaviour by using minNumVisits and removeSmallGroups.

    Function returns a list and two dictionaries. First returned element is a list
    of lists. Each element of the list represents a single pointing group, such
    that each element of that list represents an index of the ingoing table of
    all pointings.
    Dictionaries represent mappings from foldernames to pointing groups for
    ingested folders and image difference folders respectively.

    Parameters
    ----------
    pointings : VOTable
        VOTable of pointings with modified format as returned by 'read_pointings'
        function.
    raTol : float
        right ascension tolerance limits defining a group. Default: 5e-2
    decTol : float
        declination tolerance limits defining a group. Default: 5e-2
    minNumVisits : int
        if number of pointings in a group is less than minNumVisits the group is
        discarded. Default: 3
    removeSmallGroups : bool
        if True groups will be checked against minNumVisits. Default: True
    keepMaxReprocessingOnly: bool
        if True only the latest reproessing version will be kept.
    """
    groupindices = []
    ingestdict = dict()
    imdiffdict = dict()

    # we create a copy for convenience and maxIter for safety. As we are creating
    # groups we can remove all elements sorted in that group and avoid including
    # different permutations of a group later on. Max Iter is a safety issue to
    # avoid infinite while loop scenarios.
    tmpCopy = pointings.copy()
    maxIter = len(pointings)
    i = 0
    while len(tmpCopy) >= 1 and i < maxIter:
        ra = tmpCopy['ra'][0]
        dec = tmpCopy['dec'][0]

        # because we will be removing rows from tmpCopy table they will not
        # correspond to indices of the original pointings table anymore. Because
        # the tables are sorted these indice could be kept track of independently
        # but the cost of double search is not large and I wanted a sorting
        # independent function 
        raMask = np.isclose(ra,  tmpCopy['ra'],  atol=raTol)
        deMask = np.isclose(dec, tmpCopy['dec'], atol=deTol)
        indice = np.where(raMask & deMask == True)[0]

        # we will perform the second search only if the group will be kept
        # otherwise we just remove the elements and progress the search to next
        # group, this is sort of a "premature" optimization in the sense that 
        # once we remove reprocessing duplicates the group can end up smaller
        if removeSmallGroups and len(indice)<minNumVisits:
            tmpCopy.remove_rows(indice)
            continue
             
        raMask = np.isclose(ra,  pointings['ra'],  atol=raTol)
        deMask = np.isclose(dec, pointings['dec'], atol=deTol)
        allindice = np.where(raMask & deMask == True)[0]

        group = tmpCopy[indice]
        if keepMaxReprocessingOnly:
            # While reprocessing version can differ they are reprocessings of same
            # pointing and will have the same visit id. 
            allVisitIds = group["visit_id"].data.data.astype(np.int)
            uniqueVisitIds  = np.unique(allVisitIds)
            # For each unique visit in a group we find the max reprocessing version
            maxReprocessing = {u: max(group[allVisitIds==u]["reprocessing"]) \
                               for u in uniqueVisitIds}

            goodrows, badrows = [], []
            if len(uniqueVisitIds) < len(allVisitIds):
                for row, pointing in enumerate(group):
                    # for each pointing in a group we only keep the highest version
                    if pointing["reprocessing"] != maxReprocessing[int(pointing["visit_id"])]:
                        badrows.append(row)
                    else:
                        goodrows.append(row)

        # as stated above, once we clear out duplicates we again check if 
        # group is removable. We have to be careful of empty "goodrows"
        # because "indice" does not reset to empty
        if removeSmallGroups and len(goodrows)<minNumVisits and len(goodrows)>0:
            #import pdb
            #pdb.set_trace()
            tmpCopy.remove_rows(indice)
            continue
            
        # the only purpose of group is to maintain the same naming conventions
        # as Hayden so verification is easier. It is otherwise not required.
        ra, dec = group[0]["ra"], group[0]["dec"]
        ingestName = create_ingest_foldername(ra, dec)
        imdiffName = create_imdiff_foldername(ra, dec)

        # goodrows track the highest reprocessing versions. Sometimes however
        # they work out to be the only reprocessings of the visits. In those cases
        # all indices are good.
        if not goodrows:
            goodindice = allindice
        else:
            goodindice = allindice[goodrows]

        # modifying the ingest and imdiff folder names column values for good
        # rows - doing it like this avoids creating additional copies of the table
        pointings["ingest_folder"][goodindice] = [ingestName]*len(goodindice)
        pointings["imdiff_folder"][goodindice] = [imdiffName]*len(goodindice)

        ingestdict[ingestName] = goodindice
        imdiffdict[imdiffName] = goodindice
        groupindices.append(goodindice)

        tmpCopy.remove_rows(indice)
        i+=1

    return groupindices, ingestdict, imdiffdict





class PointingGroups:
    """PointingGroups facilitates working with groups of pointings without
    compromising access to all pointings or losing advance indexing capabilities.

    Class can be instantiated:
    1) by loading saved object from a file via the class  method 'load_from_file'.
    2) by sending in keyword 'table' in which case function 'group_pointings'
        will be used to sort out and recreate required mappings.
    3) by sending in 'table', 'groupIndices', 'ingestFolderMap' and 'imdiffFolderMap'
        as keywords

    The class is not consistent as slicing operations will return an astropy Table
    object. Indexing is also not guaranteed to be consistent as a view or a copy
    of the table will be returned depending on the [col, rows] or [rows, col]
    order of indices. Prefered practice is to return a copy to avoid accidental
    modification of underlying data.

    General indexing rules are that they are always with respect to groups. To
    access all pointings either index the table attribute or use getters.

    Parameters
    ----------
    table : VOTable
        modified table of all pointings as returned by 'read_pointings'. Optional.
    groupIndices : list
        list of lists containing indices of all elements of table that make a group
    ingestFolderMap : dict
        dictionary whose keys are names of folders where data was ingested.
        See attribute: ingest 
    imdiffFolderMap : dict
        dictionary whose keys are names of folders where image difference data is
        See attribute: imdiff

    Attributes
    ----------
    table : VOTable
        See parameters Table
    groups : list
        provides a map that enables integer indexing of groups of pointings to
        the table of all pointings. List of lists of indices of all elements of
        table that make a group. See parameter: groupIndices
    ingest : dict
        map between name of ingest folder and group of pointings ingested there.
        See parameter: ingestFolderMap
    imdiff : dict
        map between name of folder name containing difference images and a group
        of pointings. See parameter: imdiffFolderMap
    filename : str
        if applicable, will be set to the name of the file from which the object
        was loaded from or None if object was not created from a pickled file.
    """
    def __init__(self, **kwargs):

        tmpkeys = ["table", "groupIndices", "ingestFolderMap", "imdiffFolderMap"]
        if all(key in kwargs for key in tmpkeys):
            self._load_from_partial_init(**kwargs)
        elif "table" in kwargs:
            self._load_from_init(**kwargs)
        else:
            errmsg = ("Unknown instantiation arguments. Expected filepath "
                      "or table or \n"
                      "(table, groupIndices, ingestFolderMap, imdiffFolderMap) "
                      "but got: " + str(kwargs.keys()))
            raise TypeError(errmsg)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def _load_from_init(self, **kwargs):
        table = kwargs.pop("table")
        groupindices, ingestdict, imdiffdict = group_pointings(table, **kwargs)
        self.__set_attrs(table, groupindices, ingestdict, imdiffdict)

    def _load_from_partial_init(self, **kwargs):
        table = kwargs.pop("table")
        groupindices = kwargs.pop("groupIndices")
        ingestdict = kwargs.pop("ingestFolderMap")
        imdiffdict = kwargs.pop("imdiffFolderMap")
        self.__set_attrs(table, groupindices, ingestdict, imdiffdict)

    def __set_attrs(self, table, groups, ingest, imdiff, filepath=None):
        self.table  = table
        self.groups = groups
        self.ingest = ingest
        self.imdiff = imdiff
        self.filepath = filepath
        # flatten all groups of indices to form a mask for easier indicing later
        self._flatGroups = [index for group in groups for index in group]

    def __getitem__(self, key):
        if isinstance(key, str):
            try: return self.table[self.imdiff[key]]
            except: pass
            try: return self.table[self.ingest[key]]
            except: pass
        elif isinstance(key, int):
            return self.table[self.groups[key]]
        elif isinstance(key, slice):
            return [self.table[group] for group in self.groups[key]]
        try: return self.table[key][self._flatGroups]
        except: pass

        errmsg = "Expected key of type int or str, got {0} type {1} instead."
        raise KeyError(errmsg.format(key, type(key)))

    def __iter__(self):
        self._curr = 0
        self._max = len(self.groups)-1
        return self

    def __next__(self):
        if self._curr == self._max:
            raise StopIteration()
        self._curr += 1
        return self.table[self.groups[self._curr]]

    def write(self, filepath):
        with open(filepath, 'wb') as file:
            file.write(pickle.dumps(self))

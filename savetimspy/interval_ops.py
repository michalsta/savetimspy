from __future__ import annotations
import ncls
import numpy as np
import numpy.typing as npt


class NestedContainmentList:
    """A wrapper around ncls.FNCLS."""
    def __init__(
        self, 
        starts: npt.NDArray[float],
        stops: npt.NDArray[float],
        indices: npt.NDArray[int]|None = None,
    ):
        """ 
        Arguments:
            starts (np.array): Beginnings of the intervals to be stored in the NCLS.
            stops (np.array): Ends of the intervals to be stored in the NCLS.
            indices (np.array or Null): Optional interval indexing.
        """
        if indices is None:
            indices = np.arange(len(starts), dtype=int)
        else:
            indices = np.array(indices, dtype=int)
        self._fncls = ncls.FNCLS( 
            starts=np.array(starts, dtype=float),
            ends=np.array(stops, dtype=float),
            ids=indices,
        )

    def query(
        self,
        starts: Iterable[float],
        stops: Iterable[float],
        indices: Iterable[float]|None = None,
        _lex_sort_db_idxs: bool=True,
    ) -> pd.DataFrame:
        """Check for interval intersections with the stored intervals.

        Arguments:
            starts (np.array): Beginnings of the intervals checked against the stored ones.
            stops (np.array): Ends of the intervals checked against the stored ones.
            indices (np.array or Null): Optional interval indexing for the query intervals.

        Returns:
            tuple[np.array]: Two arrays: one with indices of the intervals in the query, and another with the indices of intervals stored in NCLS that they intersected.
        """
        if indices is None:
            indices = np.arange(len(starts), dtype=int)
        else:
            indices = np.array(indices, dtype=int)
        query_idxs, db_idxs = self._fncls.all_overlaps_both(
            np.array(starts, dtype=float),
            np.array(stops, dtype=float),
            indices,
        )
        query_idxs = query_idxs.astype(int)
        db_idxs = db_idxs.astype(int)
        if _lex_sort_db_idxs:
            db_idxs = db_idxs[np.lexsort((db_idxs, query_idxs))]
        return query_idxs, db_idxs

    def query_df(
        self,
        df: pd.DataFrame,
        _lex_sort_db_idxs: bool=True,
    ):
        """Check for interval intersections with the stored intervals.
    
        Arguments:
            df (pd.DataFrame): First column corresponds to starts, second to stops of the intervals. Each interval is checked agains intersection with all stored intervals.

        Returns:
            tuple[np.array]: Two arrays: one with indices of the intervals in the query, and another with the indices of intervals stored in NCLS that they intersected.
        """
        return self.query(
            starts=df.iloc[:,0],
            stops=df.iloc[:,1],
            indices=df.index,
            _lex_sort_db_idxs=_lex_sort_db_idxs,
        )

    @classmethod
    def from_df(cls, df):
        """Construct a nested containment list from pandas data frame.
    
        Arguments:
            df (pd.DataFrame): First column corresponds to starts, second to stops.

        Returns:
            NestedContainmentList: A nested containment list.
        """
        return cls(
            starts=df.iloc[:,0],
            stops=df.iloc[:,1],
            indices=df.index,
        )
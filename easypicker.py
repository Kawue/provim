import pandas as pd
import numpy as np
from scipy.signal import find_peaks as sp_find_peaks
from scipy.signal import peak_widths as sp_peak_width
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

class Easypicker:
    def __init__(self, dframe, aggregation, winsorize, normalize=True):
        self.dframe = dframe
        self.data = dframe.values
        self.winsorize = winsorize
        self.normalize = normalize
        self.mean_spec = np.mean(self.data, axis=0)
        #self.mean_spec = np.array([np.log(x) if x > 1 else 0 for x in np.mean(self.data, axis=0)])
        if winsorize > 0:
            if type(winsorize) != int:
                raise ValueError("winsorize has to be of type int!")
            self.winsorize_limit = sorted(self.mean_spec)[-winsorize]
            print(self.winsorize_limit)
            self.mean_spec[self.mean_spec > self.winsorize_limit] = self.winsorize_limit
        if normalize:
            mi = self.mean_spec.min()
            ma = self.mean_spec.max()
            self.mean_spec = (self.mean_spec - mi) / (ma - mi)
        #self.median_spec = np.median(self.data, axis=0)
        self.mzs = dframe.columns
        self.peaks_idx = None
        self.peaks_mzs = None
        self.peaks_dict = None
        self.deiso_peaks_idx = None
        self.deiso_peaks_mzs = None
        self.deiso_peaks_dict = None
    

    def find_peaks(self, t, rel_height=0.5):
        #peaks = detect_peaks(self.mean_spec, mph=mph, threshold=t, edge=edge, kpsh=kpsh, valley=valley)
        
        baseline = self._baseline_als(self.mean_spec)
        corr_spec = self.mean_spec - baseline
        corr_spec[corr_spec<0] = 0

        self.mean_spec = corr_spec

        self.peaks_idx, dct = sp_find_peaks(self.mean_spec, height=t)
        self.peaks_mzs = self.mzs[self.peaks_idx]
        
        # left_end, right_end are index based
        self.peaks_dict = {}
        peak_width, rel_height, left_end, right_end = sp_peak_width(self.mean_spec, self.peaks_idx, rel_height=rel_height)
        left_end = np.ceil(left_end).astype(int)
        right_end = np.floor(right_end).astype(int)
        self.peaks_dict["width"] = peak_width
        self.peaks_dict["rel_height"] = rel_height
        self.peaks_dict["left"] = left_end
        self.peaks_dict["right"] = right_end

        #return self.mzs[self.peaks_idx], self.peaks_idx
        

    
    def deisotope(self, iso_range, max_mode=True, alt_peaks_idx=None):
        def match_subprocedure(isotope_candidates, max_mode):
            isotopes_list = self._check_isotope_break(isotope_candidates, [])
            for isotopes in isotopes_list:
                try:
                    if max_mode:
                        self.deiso_peaks_idx.append(isotopes[np.argmax(self.mean_spec[isotopes])])
                    else:
                        self.deiso_peaks_idx.append(isotopes[0])
                except Exception as e:
                    print(e)

        if alt_peaks_idx is not None:
                self.peaks_idx = alt_peaks_idx

        self.deiso_peaks_idx = []
        remain = self.peaks_idx.copy()

        # Calculate all possible isotope groups
        while len(remain) > 0:
            #print()
            #print()
            #print()
            #print()
            #print()
            #print(len(remain))
            superbreak = False
            isotope_candidates = []
            to_remove = []
            # current peak and next peak
            for i, pc in enumerate(remain[:-1]):
                isotope_candidates.append(pc)
                to_remove.append(i)
                #print("-----")
                #print(isotope_candidates)
                #print(remain)
                #print("pc - i: %f - %i"%(self.mzs[pc], i))
                #print("-----")

                #das funktioniert nur einmal, bei i=1 und j=3 ist die distanz schon erreicht weil i=2 verglichen werden müsste, sofern self.mzs[pn] > self.mzs[pc] + iso_range[0] erfüllt ist

                for j, pn in enumerate(remain[i+1:], start=i+1):
                    # min dist check
                    if self.mzs[pn] > self.mzs[pc] + iso_range[0]:
                        # max dist check
                        #print("+++++")
                        #print(isotope_candidates)
                        #print(remain)
                        #print("pc - i: %f - %i"%(self.mzs[pc], i))
                        #print("min-result: %f"%(self.mzs[pc]+iso_range[0]))
                        #print("pn - j: %f - %i"%(self.mzs[pn], j))
                        #print("max-result: %f"%(self.mzs[pc]+iso_range[1]))
                        #print("+++++")
                        if self.mzs[pn] < self.mzs[pc] + iso_range[1]:
                            isotope_candidates.append(pn)
                            to_remove.append(j)
                            pc = pn
                        else:
                            #print("Too LARGE!")
                            match_subprocedure(isotope_candidates, max_mode)
                            remain = np.delete(remain, to_remove)
                            superbreak = True
                            break
                    else:
                        #print("*****")
                        #print("Too SMALL!")
                        #print(isotope_candidates)
                        #print(remain)
                        #print(len(remain))
                        #print("pc - i: %f - %i"%(self.mzs[pc], i))
                        #print("min-result: %f"%(self.mzs[pc]+iso_range[0]))
                        #print("pn - j: %f - %i"%(self.mzs[pn], j))
                        #print("*****")
                        # Append last peak of non isotope series
                        if j == len(remain[i+1:]):
                            #print("Last index j!")
                            #print(to_remove)
                            #print(remain)
                            #print(isotope_candidates)
                            match_subprocedure(isotope_candidates, max_mode)
                            remain = np.delete(remain, to_remove)
                            #print(to_remove)
                            #print(remain)
                            #print(isotope_candidates)
                            superbreak = True
                else:
                    if superbreak:
                        break
                    #print("-----")
                    #print("End of j loop")
                    #print(isotope_candidates)
                    #print(remain)
                    #print("pc - i: %f - %i"%(self.mzs[pc], i))
                    #print("-----")
                    match_subprocedure(isotope_candidates, max_mode)
                    remain = np.delete(remain, to_remove)
                    superbreak = True
                
                if superbreak:
                        break
                

            if not superbreak:
                isotope_candidates.append(remain[0])
                #print("-----")
                #print("End of i loop")
                #print(isotope_candidates)
                #print(remain)
                #print("pc - i: %f - %i"%(self.mzs[pc], i))
                #print("-----")
                to_remove.append(0)
                match_subprocedure(isotope_candidates, max_mode)
                remain = np.delete(remain, to_remove)

        self.deiso_peaks_idx.sort()
        self.deiso_peaks_mzs = self.mzs[self.deiso_peaks_idx]
        self.deiso_peaks_dict = {}
        peak_width, rel_height, left_end, right_end = sp_peak_width(self.mean_spec, self.deiso_peaks_idx)
        left_end = np.ceil(left_end).astype(int)
        right_end = np.floor(right_end).astype(int)
        self.deiso_peaks_dict["width"] = peak_width
        self.deiso_peaks_dict["rel_height"] = rel_height
        self.deiso_peaks_dict["left"] = left_end
        self.deiso_peaks_dict["right"] = right_end

    
    def create_dframe(self, deisotoped, apex_mode=False):
        if deisotoped:
            if apex_mode:
                picked_dframe = self.dframe[self.deiso_peaks_mzs]
            else:
                picked_dframe = pd.DataFrame([], index=self.dframe.index)
                mz_pairs = zip(self.deiso_peaks_dict["left"], self.deiso_peaks_dict["right"])
                for left, right in mz_pairs:
                    interval = self.mzs[left:right+1]
                    picked_dframe[np.median(interval)] = self.dframe[interval].sum(axis=1)
        else:
            if apex_mode:
                picked_dframe = self.dframe[self.peaks_mzs]
            else:
                picked_dframe = pd.DataFrame([], index=self.dframe.index)
                mz_pairs = zip(self.peaks_dict["left"], self.peaks_dict["right"])
                for left, right in mz_pairs:
                    interval = self.mzs[left:right+1]
                    picked_dframe[np.median(interval)] = self.dframe[interval].sum(axis=1)
        return picked_dframe


    def _check_isotope_break(self, isotope_pattern, isotope_list):
        # Roll to divide every value by its successor (instead of for loop)
        i = self.mean_spec[isotope_pattern]
        j = self.mean_spec[np.roll(isotope_pattern, -1)]
        l = i-j
        # Find the first local maximum (important to avoid problems with "hill structures")
        local_max_idx = np.where(l >= 0)[0][0]
        # Exclude the last index to avoid problems due to rolling
        k = i[local_max_idx:-1]-j[local_max_idx:-1]
        try:
            # Find the first local minimum after the first local maximum
            break_idx = np.where(k < 0)[0][0] + local_max_idx + 1
            #print("~~~~~")
            #print("Break at: %f - %i"%(self.mzs[isotope_pattern[break_idx]], break_idx))
            #print("~~~~~")
            # Fix the first isotope series
            isotope_list.append(isotope_pattern[:break_idx])
            # Add the remaining isotopes into recursive break detection
            self._check_isotope_break(isotope_pattern[break_idx:], isotope_list)
            return isotope_list
        except Exception as e:
            #print("No Break!")
            if isinstance(e, IndexError):
                isotope_list.append(isotope_pattern)
                return isotope_list
            else:
                print("Exception in method check_isotope_break() of class Easypicker. Exception type: %s: %s"%(type(e).__name__, e))


    # Baseline Correction with Asymmetric Least Squares Smoothing
    def _baseline_als(self, y, lam=10**5, p=0.01, niter=10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    

    '''
    def winsorizing(spectrum, isotope_idx_list, divide=5):
        peaks = self.peaks_dict
        spectrum_cpy = np.copy(spectrum)
        if len(self.peaks_idx) == 0:
            return spectrum_cpy
        
        if divide >= len(indices):
            divide = len(indices) - 1

        indices, widths, lefts, rights = zip(*sorted(zip(self.peaks_idx, peaks["width"], peaks["left"], peaks["right"]),
                                                        key=lambda x: spectrum[x[0]], reverse=True))


        lefts = list(lefts)
        rights = list(rights)
        used_isotopes = set()
        permited_max_value = spectrum[indices[divide]]
        for i in range(0, divide):
            if spectrum[indices[i]] in used_isotopes:
                divide += 1
                continue
            found_left = found_right = False
            for j in self.deiso_peaks_idx:
                if j[0] <= indices[i] < j[-1]:
                    lefts[i] = j[0]
                    rights[i] = j[-1]
                    used_isotopes.update(spectrum[lefts[i]:rights[i]])
                    break
                if j[0] > indices[i]:
                    break
            for n, k in enumerate(indices):
                if k == lefts[i]:
                    lefts[i] = lefts[n]
                    found_left = True
                    if found_left and found_right:
                        break
                elif k == rights[i]:
                    rights[i] = rights[n]
                    found_right = True
                    if found_left and found_right:
                        break
            shrink_factor = spectrum[indices[i]] / permited_max_value
            spectrum_cpy[lefts[i]:rights[i]] /= shrink_factor
        return spectrum_cpy
    '''
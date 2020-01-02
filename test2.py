def deisotope(self, iso_range, max_mode=True):
    self.deiso_peaks_idx = []
    remain = self.peaks_idx.copy()
    isotopes_dict = {}
    isotopes_key = 0
    # Calculate all possible isotope groups
    while remain:
        isotopes = []
        # current peak and last peak
        # roll will result in omitting the first peak
        for pl, pc in zip(np.roll(remain, 1), remain):
            if self.mzs[pc] > self.mzs[pl] + iso_range[0]:
                hier muss  theoretisch pl mit dem nächsten pc verglichen werden, weil dieser die reihe theoretisch forführen kann.
                doppel for notwendig?
                if self.mzs[pc] < self.mzs[pl] + iso_range[1]:
                    isotopes.append(pc)
                else:
                    isotopes_dict[isotopes_key] = isotopes
                    isotopes_key += 1
                    isotopes = [pc]
    
        remove = np.array([x for _, x in isotopes_dict.items()]).flatten()
        remain = np.array(list(set(remain) - set(remove)))




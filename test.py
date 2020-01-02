def deisotope(self, iso_range, max_mode=True, alt_peaks_idx=None):
        if alt_peaks_idx is not None:
            self.peaks_idx = alt_peaks_idx

        # current peak and last peak
        self.deiso_peaks_idx = []
        lpeaks_idx = np.roll(self.peaks_idx, 1)
        isotopes = []
        process_later = {}
        # roll will result in omitting the first peak
        iso_key = 0
        for pl, pc in zip(lpeaks_idx, self.peaks_idx):
            if self.mzs[pl] + iso_range < self.mzs[pc]:
                if self.mzs[pl] + iso_range[0] < self.mzs[pc]:
                    process_later[iso_key] = pc
                    iso_key += 1
                else:
                    isotope_list = self._check_isotope_break(isotopes, [])
                    for isotopes in isotope_list:
                        if len(isotope_list) > 1:
                            plt.plot(self.mzs[isotopes[-1]], 0.2, "y*")
                        try:
                            if max_mode:
                                self.deiso_peaks_idx.append(isotopes[np.argmax(self.mean_spec[isotopes])])
                            else:
                                self.deiso_peaks_idx.append(isotopes[0])
                            isotopes = [pc]
                        except Exception as e:
                            print(e)
            else:
                isotopes.append(pc)
                
        else:
            # Adds final mz value
            isotope_list = self._check_isotope_break(isotopes, [])
            for isotopes in isotope_list:
                plt.plot(self.mzs[isotopes[-1]], 0.2, "g*")
                if max_mode:
                    self.deiso_peaks_idx.append(isotopes[np.argmax(self.mean_spec[isotopes])])
                else:
                    self.deiso_peaks_idx.append(isotopes[0])
from .base import *


class GodasDataset(BaseDataset):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.input_var_list = None
        self.times = None

        self.input_var_list = self.list_dir_fn(self.root)
        if len(args.input_var_list) != 0:
            for v in args.input_var_list:
                if v not in CMIP62GODAS:
                    raise ValueError(f'variable {v} is not contained in CMIP6')
            self.input_var_list = args.input_var_list

        if len(args.var_list_not_used) != 0:
            for v in args.var_list_not_used:
                if v in self.input_var_list:
                    self.input_var_list.remove(v)

        self.input_var_list = sorted(self.input_var_list)

        if 'tauu' in self.input_var_list:
            self.input_var_list.remove('tauu')
        if 'tauv' in self.input_var_list:
            self.input_var_list.remove('tauv')
        
        self.single_lev_vars = [v for v in self.input_var_list if v in SINGLE_LEVEL_VARS]
        self.multi_lev_vars = [v for v in self.input_var_list if v in MULTY_LEVEL_VARS]
        self.atmo_var_list = sorted(args.atmo_var_list)

        self.mix = len(self.single_lev_vars) != 0 and len(self.multi_lev_vars) != 0

        self.load_times()

        self.num_times = len(self.times)
        self.num_var = len(self.input_var_list)
        self.input_steps = args.input_steps
        self.predict_steps = args.predict_steps

        self.num_samples = self.num_times - self.input_steps - self.predict_steps + 1

        self.split_by = args.train_val_test_split_by
        if self.split_by == 'time':
            self.specify_split_by_time()
        else:
            ...

        if args.do_cache:
            self.cache_samples()

    def get_input_var_list_cmip6(self):
        return self.input_var_list

    def get_input_var_list_godas(self):
        return [CMIP62GODAS[v] for v in self.input_var_list]


    def get_var_index(self, v_name):
        if self.mix:
            if v_name in self.single_lev_vars:
                return self.single_lev_vars.index(v_name)
            else:
                return self.multi_lev_vars.index(v_name)

        return self.input_var_list.index(v_name)

    def get_subset_by_years(self, start_year, end_year=None):
        # subset for predicting values in [start_year, end_year]
        start_idx, end_idx = None, None
        for i in range(self.num_times):
            year = int(self.times[i].split('_')[0])
            if year == start_year:
                start_idx = i - self.input_steps
            elif end_year is not None and year == end_year:
                end_idx = i + 11
        if start_idx < 0 or (end_idx and end_idx > len(self)):
            raise ValueError("invalid start year/ end year")
        if end_idx:
            return self.get_subset(list(range(start_idx, end_idx)))
        else:
            return self.get_subset(list(range(start_idx, len(self))))

    def get_train_dataset(self):
        indices = np.array(range((self.train_val_split_year - self.start_year + 1)
                        * 12 - self.input_steps - self.predict_steps + 1))
        return self.get_subset(indices.tolist())

    def get_val_dataset(self):
        if self.val_test_split_year is None:
            raise ValueError("The val_test_split_year is None")
        indices = np.array(range((self.train_val_split_year - self.start_year + 1) * 12,
                                (self.val_test_split_year - self.start_year + 1) * 12 - self.input_steps - self.predict_steps + 1))
        return self.get_subset(indices.tolist())
    
    def get_test_dataset(self):
        if self.val_test_split_year is None:
            raise ValueError("The val_test_split_year is None")
        indices = np.array(range((self.val_test_split_year - self.start_year + 1) * 12,
                                self.num_times - self.input_steps - self.predict_steps + 1))
        return self.get_subset(indices.tolist())

    def get_ocean_vars(self, base_path, time_range):
        data = []
        for v in self.input_var_list:
            if v in self.single_lev_vars:
                combine_fn = torch.stack
            else:
                combine_fn = torch.cat
            data.append(combine_fn([
                self.get_data(os.path.join(base_path, v, f'{self.times[i]}.npy'))
                for i in time_range
            ]))
        return torch.cat(data).float()
    
    def get_label_values(self, base_path, time_range):
        data = [torch.cat([
            self.get_data(os.path.join(base_path, v, f'{self.times[i]}.npy')).unsqueeze(0)
            if v in self.single_lev_vars else self.get_data(os.path.join(base_path, v, f'{self.times[i]}.npy'))
            for v in self.input_var_list
        ]) for i in time_range]
        if len(data) == 1:
            return data[0]
        else:
            return torch.stack(data).float()

    def get_data_atmo(self, path):
        if not os.path.exists(path):
            assert 0
    
        data = np.load(path)
        return torch.from_numpy(data).float()

    def get_atmo_vars(self, base_path, time_range):
        data = [torch.cat([
            self.get_data_atmo(os.path.join(base_path, v, f'{self.times[i]}.npy')).unsqueeze(0)
            for i in time_range
        ]) for v in self.atmo_var_list]
        return torch.cat(data).float()

    def __getitem__(self, index):

        # if self.cached:
        #     return torch.load(os.path.join(self.args.cache_sample_dir, f'{index}.pt'))

        start_month = int(self.times[index].split('_')[-1])
        
        inputs_range = range(index, index+self.input_steps)
        labels_range = range(index+self.input_steps, index+self.input_steps+self.predict_steps)

        ocean_vars = self.get_ocean_vars(self.root, inputs_range)
        labels = self.get_label_values(self.root, labels_range)
        atmo_vars = self.get_atmo_vars(self.root, inputs_range)

        ret = {
            'ocean_vars': ocean_vars,
            'atmo_vars': atmo_vars,
            'labels': labels,
            'start_month': torch.tensor(start_month)
        }

        return ret

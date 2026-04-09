from .base import *


class Cmip6SingleDataset(BaseDataset):
    def __init__(self, args, root=None, split='train') -> None:
        super().__init__(args)

        if root is not None:
            self.root = root
        else:
            self.root = args.data_dir

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


        self.load_times(all_time=True)

        self.num_times = len(self.times)
        self.num_var = len(self.input_var_list)

        self.input_steps = args.input_steps
        self.predict_steps = args.predict_steps

        self.max_t = args.max_t
        self.split = split

        self.num_samples = self.num_times - self.input_steps - self.predict_steps - (self.max_t-1) + 1

        if args.lead_t_weight:
            t_weight = np.linspace(args.max_t_weight, args.min_t_weight, self.max_t)
            self.t_weight = t_weight / t_weight.sum()
        else:
            self.t_weight = np.ones(self.max_t) / self.max_t

        self.split_by = args.train_val_test_split_by

        if self.split_by == 'time':
            assert 0
            self.specify_split_by_time()
        else:
            ...

        if args.do_cache:
            self.cache_samples()

    def random_lead_t(self):
        # print(self.max_t, self.t_weight.shape)
        return int(np.random.choice(np.arange(self.max_t), size=1, p=self.t_weight))

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
        data = np.load(path)
        return torch.from_numpy(data)
    
    def get_atmo_vars(self, base_path, time_range):
        data = [torch.cat([
            self.get_data_atmo(os.path.join(base_path, v, f'{self.times[i]}.npy')).unsqueeze(0)
            for i in time_range
        ]) for v in self.atmo_var_list]
        return torch.cat(data).float()
        
    def __len__(self):
        if self.split == 'train':
            return self.max_t * self.num_samples
        else:
            return self.num_samples

    def __getitem__(self, index):

        if self.split == 'train':
            lead_t = index % self.max_t
            index = index // self.max_t
        else:
            lead_t = self.max_t - 1

        start_month = int(self.times[index].split('_')[-1])

        # lead_t = self.random_lead_t()
        inputs_range = range(index, index+self.input_steps)
        labels_range = range(index+self.input_steps+lead_t, index+self.input_steps+lead_t+self.predict_steps)

        ocean_vars = self.get_ocean_vars(self.root, inputs_range)
        atmo_vars = self.get_atmo_vars(self.root, inputs_range)
        labels = self.get_label_values(self.root, labels_range)

        return {
            'ocean_vars': ocean_vars,
            'atmo_vars': atmo_vars,
            'labels': labels,
            'lead_time': torch.tensor(lead_t),
            'start_month': torch.tensor(start_month)
        }


class Cmip6Dataset(BaseDataset):
    def __init__(self, args, root=None, split='train') -> None:
        super().__init__(args)

        if root is not None:
            self.root = root
        else:
            self.root = args.data_dir

        self.source_ids = None
        self.input_var_list = None
        self.times = None
        self.split = split

        self.source_ids = None
        self.init_source_id()

        self.max_t = args.max_t
        self.datasets = [Cmip6SingleDataset(args, os.path.join(self.root, sid), split) for sid in self.source_ids]

        self.input_var_list = self.datasets[0].input_var_list
        self.times = self.datasets[0].times
        self.num_times = self.datasets[0].num_times
        self.num_samples_per_sid = len(self.datasets[0])
        self.num_samples = self.num_samples_per_sid * len(self.source_ids)


        for i in range(1, len(self.datasets)):
            if self.datasets[i].input_var_list != self.input_var_list:
                raise ValueError(f"prepare data error, inconsistency of vars, check sid {self.source_ids[i]}")
            if self.datasets[i].num_times != self.num_times:
                raise ValueError(f"prepare data error, inconsistency of times, check sid {self.source_ids[i]}")

        self.single_lev_vars = self.datasets[0].single_lev_vars
        self.multi_lev_vars = self.datasets[0].multi_lev_vars

        self.mix = self.datasets[0].mix

        self.num_var = len(self.input_var_list)
        self.input_steps = args.input_steps
        self.predict_steps = args.predict_steps

        if args.do_cache:
            self.cache_samples()

    def init_source_id(self) -> None:
        source_ids = self.list_dir_fn(self.root)
        for s in self.args.model_list_not_used:
            if s in source_ids:
                source_ids.remove(s)
        val_model_list = self.args.val_model_list
        test_model_list = self.args.test_model_list
        val_test_list = val_model_list + test_model_list
        for p in val_test_list:
            if p not in source_ids:
                raise ValueError(f"model {p} not included in data dir")

        if self.split == 'train':
            if len(self.args.train_model_list) == 0:
                self.source_ids = [
                    p for p in source_ids if p not in val_test_list]
            else:
                self.source_ids = self.args.train_model_list
        elif self.split == 'valid':
            self.source_ids = val_model_list
        elif self.split == 'test':
            self.source_ids = test_model_list

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

    def specify_split_by_model(self):
        self.val_model_list = self.args.val_model_list
        self.test_model_list = self.args.test_model_list
        val_test_list = self.val_model_list + self.test_model_list
        for p in val_test_list:
            if p not in self.source_ids:
                raise ValueError(f"model {p} not included in data dir")
        if len(self.args.train_model_list) == 0:
            self.train_model_list = [
                p for p in self.source_ids if p not in val_test_list]
        else:
            self.train_model_list = self.args.train_model_list

        if len(set(self.train_model_list) & set(self.val_model_list)) != 0 or \
                len(set(self.train_model_list) & set(self.test_model_list)) != 0:
            raise ValueError(
                f"there exists some models included in both train and val/test list.")

    def get_train_dataset(self):
        if self.split_by == 'model':
            indices = []
            for p in self.train_model_list:
                sid_idx = self.source_ids.index(p)
                indices.extend(
                    list(range(self.num_samples_per_sid*sid_idx, self.num_samples_per_sid*(sid_idx+1)))
                )
        else:
            assert 0

        return self.get_subset(indices, train=True)

    def get_val_dataset(self):
        if self.split_by == 'model':
            if len(self.val_model_list) == 0:
                raise ValueError("The val_model_list is empty")
            indices = []
            for p in self.val_model_list:
                sid_idx = self.source_ids.index(p)
                indices.extend(
                    list(range(self.num_samples_per_sid*sid_idx, self.num_samples_per_sid*(sid_idx+1)))
                )
        else:
            assert 0
        return self.get_subset(indices, train=False)

    def get_test_dataset(self):
        if self.split_by == 'model':
            if len(self.test_model_list) == 0:
                raise ValueError("The test_model_list is empty")
            indices = []
            for p in self.test_model_list:
                sid_idx = self.source_ids.index(p)
                indices.extend(
                    list(range(self.num_samples_per_sid*sid_idx, self.num_samples_per_sid*(sid_idx+1)))
                )
        else:
            assert 0
        return self.get_subset(indices, train=False)

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
            return torch.stack(data)

    def __getitem__(self, index):

        if self.cached:
            return torch.load(os.path.join(self.args.cache_sample_dir, f'{index}.pt'))

        s_idx = index // self.num_samples_per_sid
        t_idx = index % self.num_samples_per_sid
        ret = self.datasets[s_idx][t_idx]

        return ret

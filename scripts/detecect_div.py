

file = open(rf_model, "rb")
rf_cls = pickle.load(file)
def detect_div(prop_df)
temp.create_candidate_pairs()
prop_all = create_pairs_all_props_list(temp.props_data, temp.all_comb, 'na', output_list=[])
prop_df = create_all_props_df(prop_all)

file = open(rf_model, "rb")
rf_cls = pickle.load(file)
prop_df['prediction'] = rf_cls.predict(prop_df[features])
prop_df['prob'] = None
prop_df['prob'] = rf_cls.predict_proba(prop_df[features])
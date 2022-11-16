# run this cell
# formerly ID
data_subjects = sy.DataSubjectArray.from_objs(dataset["UserID"])

# formerly age_data, Age
rating_data = sy.Tensor(dataset["rating"]).annotate_with_dp_metadata(
   1, 5, data_subjects=data_subjects
)
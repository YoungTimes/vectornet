from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

##set root_dir to the correct path to your dataset folder
root_dir = '../../vectornet/data/forecasting_sample/data/'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:',len(afl))

am = ArgoverseMap()

print(vars(afl[0]))

# for argoverse_forecasting_data in (afl):
#     print(argoverse_forecasting_data.agent_traj)


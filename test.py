import Models.VGG_16_PT
import parameter

model = Models.VGG_16_PT.Model(parameter.numclasses)

model.fit()
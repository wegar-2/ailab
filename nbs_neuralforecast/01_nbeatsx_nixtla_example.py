"""
Example of fitting NBEATSx in nbs_neuralforecast
Source: https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html
"""

import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = AirPassengersPanel[
        AirPassengersPanel.ds < AirPassengersPanel['ds'].values[-12]]
    y_test = AirPassengersPanel[
        AirPassengersPanel.ds >= AirPassengersPanel['ds'].values[-12]
    ].reset_index(drop=True)
    return y_train, y_test


if __name__ == "__main__":

    y_train, y_test = get_data()

    model = NBEATSx(
        h=12,
        input_size=24,
        loss=DistributionLoss(distribution='Normal', level=[80, 90]),
        scaler_type='robust',
        dropout_prob_theta=0.5,
        stat_exog_list=['airline1'],
        futr_exog_list=['trend'],
        max_steps=200,
        val_check_steps=10,
        early_stop_patience_steps=2
    )

    nf = NeuralForecast(models=[model], freq='M')
    nf.fit(df=y_train, static_df=AirPassengersStatic, val_size=12)
    y_hat = nf.predict(futr_df=y_test)

    # Plot quantile predictions
    y_hat = y_hat.reset_index(drop=False).drop(columns=['unique_id', 'ds'])
    plot_df = pd.concat([y_test, y_hat], axis=1)
    plot_df = pd.concat([y_train, plot_df])

    plot_df = (plot_df[plot_df.unique_id == 'Airline1'].
               drop('unique_id', axis=1))
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['NBEATSx'], c='purple', label='mean')
    plt.plot(
        plot_df['ds'], plot_df['NBEATSx-median'], c='blue', label='median')
    plt.fill_between(
        x=plot_df['ds'][-12:],
        y1=plot_df['NBEATSx-lo-90'][-12:].values,
        y2=plot_df['NBEATSx-hi-90'][-12:].values,
        alpha=0.4, label='level 90'
    )
    plt.legend()
    plt.grid()
    plt.plot()

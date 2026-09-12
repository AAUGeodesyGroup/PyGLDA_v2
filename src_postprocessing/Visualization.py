from src_DA.configure_DA import config_DA
from src_postprocessing.statistical_analysis import BasinAverageAnalysis_post
from src_DA.EnumDA import WaterGap_storage_variables, Stage
from pathlib import Path
import numpy as np



class visualization:

    def __init__(self, configDA:config_DA):
        self._config = configDA
        pass

    def basin_ensemble(self, allow_pop_up: bool = True, fig_path=None):
        # Implement visualization logic here
        import pygmt


        cg = self._config

        ens_size = cg.basic.ensemble

        bp = BasinAverageAnalysis_post(ens=cg.basic.ensemble, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01',
                                       date_end='2000-01-31')

        stage = Stage.DA
        states = bp.load_states(load_dir=Path(cg.basic.res_permanent)/cg.basic.case, prefix=stage.name)

        fan_time = states['time']

        '''plot figure'''
        fig = pygmt.Figure()
        i = 0
        for state in WaterGap_storage_variables:
            i += 1

            vv = states['basin'][state.name][0]

            vmin, vmax = np.min(vv[20:]), np.max(vv[20:])
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.1
            if dmax <10:
                continue
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            if i == 6:
                fig.shift_origin(yshift='22c', xshift='14c')
                pass
            pygmt.config(FONT_TITLE="19p,5", MAP_TITLE_OFFSET="-0.2p", MAP_FRAME_TYPE="plain",
                         FONT_ANNOT_PRIMARY='11p,5', FONT_LABEL='11p,5', MAP_TICK_LENGTH='7p')

            if len(fan_time) > 720:
                fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % state.name, "xa2f1", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])
            else:
                fig.basemap(region=[fan_time[0] - 0.2, fan_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % state.name, "xa1f0.5", 'ya%df%d+lwater [mm]' % (sp_2, sp_1)])

            mean = None
            for ens in reversed(range(ens_size + 1)):
                vv = states['basin'][state.name][ens]

                if ens == 0:
                    # fig.plot(x=fan_time, y=vv, pen="1p,blue", label='%s' % (state), transparency=30)
                    pass
                else:
                    fig.plot(x=fan_time, y=vv, pen="1p,grey")
                    if mean is None:
                        mean = vv
                    else:
                        mean += vv

            fig.plot(x=fan_time, y=mean / ens_size, pen="1.5p,blue", label='Mean', transparency=30)
            fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.shift_origin(yshift='-4.4c')

        fig_path = Path(fig_path)/'figures'
        Path(fig_path).mkdir(parents=False, exist_ok=True)
        fig.savefig(str(Path(fig_path) / 'Components.pdf'))
        fig.savefig(str(Path(fig_path) / 'Components.png'))
        if allow_pop_up:
            fig.show()
        pass


    def GRACE_OL_DA(self, allow_pop_up: bool = True, fig_path=None, signal=WaterGap_storage_variables.tws.name):
        # Implement visualization logic here
        import pygmt


        cg = self._config
        basin = cg.basic.basin

        bp = BasinAverageAnalysis_post(ens=cg.basic.ensemble, case=cg.basic.case, basin=cg.basic.basin,
                                       date_begin='2000-01-01',
                                       date_end='2000-01-31')

        stage = Stage.OL
        states_OL = bp.load_states(load_dir=Path(cg.basic.res_permanent)/cg.basic.case, prefix=stage.name)

        stage = Stage.DA
        states_DA = bp.load_states(load_dir=Path(cg.basic.res_permanent) / cg.basic.case, prefix=stage.name)

        '''load GRACE'''
        # dp_dir = Path(setting_dir) / 'DA_setting.json'
        # dp4 = json.load(open(dp_dir, 'r'))
        # GRACE = pp.get_GRACE(obs_dir=dp4['obs']['dir'])
        GRACE = bp.load_GRACE(prefix=cg.basic.basin, load_dir=Path(cg.basic.res_permanent) / cg.basic.case)

        OL_time = states_OL['time']
        DA_time = states_DA['time']
        GR_time = GRACE['time']

        '''plot figure'''
        fig = pygmt.Figure()

        '''plot figure'''
        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="17p,5", MAP_TITLE_OFFSET="0p", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY='10p,5',
                     FONT_LABEL='10p,5', MAP_TICK_LENGTH='7p')

        i = 0
        offset = 12
        height = 4.6

        keys = list(GRACE['ens_mean'].keys())

        for j in range(len(keys)):
            i += 1
            basin_id = 'basin_%s' % j
            GRACE_ens_mean = GRACE['ens_mean'][basin_id]
            GRACE_original = GRACE['original'][basin_id]

            if basin_id == 'basin_0':
                basin_id='basin'
            else:
                basin_id = 'sub_'+basin_id
            OL = states_OL[basin_id][signal][0]
            OL_ens_mean = np.mean(np.array(list(states_OL[basin_id][signal].values()))[1:, ], axis=0)
            DA_ens_mean = np.mean(np.array(list(states_DA[basin_id][signal].values()))[1:, ], axis=0)

            values = [GRACE_ens_mean, OL, OL_ens_mean, DA_ens_mean]
            vvmin = []
            vvmax = []
            for vv in values:
                vvmin.append(np.min(vv[5:]))
                vvmax.append(np.max(vv[5:]))

            vmin, vmax = min(vvmin), min(vvmax)
            dmin = vmin - (vmax - vmin) * 0.1
            dmax = vmax + (vmax - vmin) * 0.15
            sp_1 = int(np.round((vmax - vmin) / 10))
            if sp_1 == 0:
                sp_1 = 0.5
            sp_2 = sp_1 * 2

            if len(OL_time) > 365 * 4:
                fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa2f1g",
                                   'ya%df%dg+lwater [mm]' % (sp_2, sp_1)])
            else:
                fig.basemap(region=[OL_time[0] - 0.2, OL_time[-1] + 0.2, dmin, dmax], projection='X12c/3c',
                            frame=["WSne+t%s" % (basin + '_' + signal + '_' + basin_id), "xa1f0.5g",
                                   'ya%df%dg+lwater [mm]' % (sp_2, sp_1)])

            # fig.plot(x=OL_time, y=OL, pen="0.5p,blue,-", label='%s' % ('OL_unperturbed'), transparency=30)
            fig.plot(x=OL_time, y=OL_ens_mean, pen="1.0p,blue,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL, pen="0.5p,grey,-", label='%s' % ('OL'), transparency=30)
            # fig.plot(x=OL_time, y=OL_ens_mean, pen="0.5p,red", label='%s' % ('OL_ens_mean'), transparency=30)
            fig.plot(x=DA_time, y=DA_ens_mean, pen="1.0p,green", label='%s' % ('DA'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_ens_mean, pen="0.5p,black", label='%s' % ('GRACE_ens_mean'), transparency=30)
            # fig.plot(x=GR_time, y=GRACE_original, pen="0.5p,purple,--.", label='%s' % ('GRACE_original'),
            #          transparency=30)
            fig.plot(x=GR_time, y=GRACE_original, style="c.2c", fill="black", label='%s' % ('GRACE'), transparency=30)

            # fig.legend(position='jTR', box='+gwhite+p0.5p')
            fig.legend(position='jBL')

            sf = '%sc' % ((offset - 1) * height)
            if i % offset == 0:
                fig.shift_origin(yshift=sf, xshift='14c')
                continue

            fig.shift_origin(yshift='-%sc' % height)

            pass

        # fig_postfix = '0'
        fig_path = Path(fig_path) / 'figures'
        fig.savefig(str(fig_path / ('DA_%s.pdf' % signal)))
        fig.savefig(str(fig_path / ('DA_%s.png' % signal)))

        if allow_pop_up:
            fig.show()
        pass
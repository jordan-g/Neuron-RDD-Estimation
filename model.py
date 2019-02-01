import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

# PARAMETERS
dt               = 0.001             # timestep (s)
v_reset          = 0                 # reset voltage
v_threshold      = 1                 # spike threshold
tau              = 0.01              # synaptic time constant (s) - used to calculate voltage
refractory_time  = np.ceil(0.003/dt) # refractory time (timesteps)
u_window         = 1                 # input drive window used to determine near-threshold data points for RDD
tau_s            = 0.003             # synaptic time constant (s) - used to calculate input
tau_L            = 0.01              # leak time constant (s) - used to calculate input
mem              = int(20/dt)        # spike history memory length (timesteps)
RDD_window       = 0.02/dt           # RDD integration window length (timesteps)
RDD_init_window  = 0.1               # window around spike threshold that determines when an RDD integration window is initiated

# whether to show a live updating plot of weights vs. estimated RDD values
updating_plot = True

# KERNEL FUNCTION
def kappa(x):
    return (np.exp(-x/(tau_L/dt)) - np.exp(-x/(tau_s/dt)))/((tau_L/dt) - (tau_s/dt))

def get_kappas(n=mem):
    return np.array([kappa(i+1) for i in range(n)])

kappas = np.flipud(get_kappas(mem))[:, np.newaxis] # initialize kappas array

class Layer():
    def __init__(self, size):
        self.size                 = size                                      # number of units
        self.u                    = v_reset*np.ones((self.size, 1))           # input drives
        self.max_u                = np.zeros((self.size, 1))                  # maximum input drives for the current RDD integration windows
        self.v                    = v_reset*np.ones((self.size, 1))           # voltages
        self.dv_dt                = np.zeros((self.size, 1))                  # changes in voltages
        self.fired                = np.zeros((self.size, 1)).astype(bool)     # whether units have spiked
        self.refractory_time_left = np.zeros((self.size, 1))                  # time left in the refractory periods of units
        self.RDD_time_left        = np.zeros((self.size, 1))                  # time left in RDD integration windows of units
        self.spike_hist           = np.zeros((self.size, mem), dtype=np.int8) # spike histories of all units
        self.feedback             = np.zeros((self.size, 1))                  # feedback input arriving at each unit
        self.R                    = np.zeros((self.size, 1))                  # reward calculated in RDD integration windows
        self.n_spikes             = np.zeros((self.size, 1))                  # number of spikes during RDD integration windows

        # recording lists
        self.max_drives_below     = [ [] for j in range(self.size) ]
        self.max_drives_above     = [ [] for j in range(self.size) ]
        self.Rs_below             = [ [] for j in range(self.size) ]
        self.Rs_above             = [ [] for j in range(self.size) ]

        # initialize lists of which (if any) neurons are updating their RDD estimate and are below or above threshold
        self.neurons_updated_above = []
        self.neurons_updated_below = []

        # RDD estimation parameters
        self.RDD_params        = np.ones((self.size, 4)) # list of RDD estimation parameters - c_r, c_l, m_r and m_l
        self.RDD_params[:, 2:] = 0                       # initialize y-intercepts to 1, slopes to 0
        self.eta               = 0.001                   # learning rate

    def update(self, I, feedback, time=0):
        self.feedback = feedback

        # determine which neurons are just ending their RDD integration window
        self.RDD_window_ending_mask = self.RDD_time_left == 1

        # update maximum input drives for units in their RDD integration window
        self.max_u[np.logical_and(self.RDD_time_left > 0, self.u > self.max_u)] = self.u[np.logical_and(self.RDD_time_left > 0, self.u > self.max_u)]

        # update rewards for units in their RDD integration window
        self.R[self.RDD_time_left > 0] += self.feedback[self.RDD_time_left > 0]

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.refractory_time_left > 0] -= 1

        # calculate changes in voltages and input drives, and update both
        self.dv_dt    = -self.v/(tau) + 2*(I - self.v)
        self.u       += dt*self.dv_dt
        self.v       += dt*self.dv_dt

        # determine which neurons are in a refractory period
        refractory_mask = self.refractory_time_left > 0

        # determine which neurons are above spiking threshold
        threshold_mask  = self.v >= v_threshold

        # neurons above threshold that are not in their refractory period will spike
        self.fired *= False
        self.fired[np.logical_and(threshold_mask, refractory_mask == False)] = True

        # reset voltages of neurons that spiked
        self.v[self.fired] = v_reset

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.fired] = refractory_time

        # update RDD estimates (only neurons whose RDD integration window has ended will update their estimate)
        self.update_RDD_estimate(time=time)

        # decrement time left in RDD integration windows
        self.RDD_time_left[self.RDD_time_left > 0] -= 1

        # reset the input drive to match the voltage, for neurons that are not in an RDD integration window
        self.u[self.RDD_time_left == 0] = self.v[self.RDD_time_left == 0]

        # determine which neurons are starting a new RDD integration window
        self.new_RDD_window_mask = np.logical_and(np.abs(v_threshold - self.u) <= RDD_init_window, self.RDD_time_left == 0)

        self.RDD_time_left[self.new_RDD_window_mask] = RDD_window

        # update number of spikes that have occurred during RDD integration windows
        self.n_spikes[np.logical_and(self.RDD_time_left > 0, self.fired)] += 1

        # update spike histories
        self.spike_hist = np.concatenate([self.spike_hist[:, 1:], self.fired.astype(int)], axis=1)

    def update_RDD_estimate(self, time=0):
        # reset lists of which (if any) neurons are updating their RDD estimate and are below or above threshold
        self.neurons_updated_above = []
        self.neurons_updated_below = []

        # figure out which neurons are at the end of their RDD integration window, and either just spiked or almost spiked
        just_spiked_mask   = np.logical_and(self.RDD_window_ending_mask, np.logical_and(np.abs(self.max_u - v_threshold) <= u_window, self.n_spikes >= 1))[:, 0]
        almost_spiked_mask = np.logical_and(self.RDD_window_ending_mask, np.logical_and(np.abs(self.max_u - v_threshold) <= u_window, self.n_spikes < 1))[:, 0]

        # update RDD estimates for neurons that just spiked or almost spiked
        if np.sum(just_spiked_mask) > 0:
            err = self.RDD_params[just_spiked_mask, 2]*self.max_u[just_spiked_mask, 0] + self.RDD_params[just_spiked_mask, 0] - self.R[just_spiked_mask, 0]
            self.RDD_params[just_spiked_mask, 2] -= self.eta*np.squeeze(err*self.max_u[just_spiked_mask, 0])
            self.RDD_params[just_spiked_mask, 0] -= self.eta*np.squeeze(err)

            # update list of which (if any) neurons are updating their RDD estimate and are above threshold
            self.neurons_updated_above = list(np.where(just_spiked_mask))
        if np.sum(almost_spiked_mask) > 0:
            err = self.RDD_params[almost_spiked_mask, 3]*self.max_u[almost_spiked_mask, 0] + self.RDD_params[almost_spiked_mask, 1] - self.R[almost_spiked_mask, 0]
            self.RDD_params[almost_spiked_mask, 3] -= self.eta*np.squeeze(err*self.max_u[almost_spiked_mask, 0])
            self.RDD_params[almost_spiked_mask, 1] -= self.eta*np.squeeze(err)

            # update list of which (if any) neurons are updating their RDD estimate and are below threshold
            self.neurons_updated_below = list(np.where(almost_spiked_mask))

    def reset_RDD_variables(self):
        # reset RDD variables for neurons whose RDD integration window has ended
        self.n_spikes[self.RDD_window_ending_mask] = 0
        self.R[self.RDD_window_ending_mask]        = 0
        self.max_u[self.RDD_window_ending_mask]    = 0

class FinalLayer():
    def __init__(self, size):
        self.size                 = size                                      # number of units
        self.v                    = v_reset*np.ones((self.size, 1))           # voltages
        self.dv_dt                = np.zeros((self.size, 1))                  # changes in voltages
        self.fired                = np.zeros((self.size, 1)).astype(bool)     # whether units have spiked
        self.refractory_time_left = np.zeros((self.size, 1))                  # time left in the refractory periods of units
        self.spike_hist           = np.zeros((self.size, mem), dtype=np.int8) # spike histories of all units

    def update(self, I, time=0):
        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.refractory_time_left > 0] -= 1

        # calculate changes in voltages and input drives, and update both
        self.dv_dt    = -self.v/(tau) + 2*(I - self.v)
        self.v       += dt*self.dv_dt

        # determine which neurons are in a refractory period
        refractory_mask = self.refractory_time_left > 0

        # determine which neurons are above spiking threshold
        threshold_mask  = self.v >= v_threshold

        # neurons above threshold that are not in their refractory period will spike
        self.fired *= False
        self.fired[np.logical_and(threshold_mask, refractory_mask == False)] = True

        # reset voltages of neurons that spiked
        self.v[self.fired] = v_reset

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.fired] = refractory_time

        # update spike histories
        self.spike_hist = np.concatenate([self.spike_hist[:, 1:], self.fired.astype(int)], axis=1)

# define a custom function to replace plt.pause() -- this function prevents the window from being brought to the front every time it is called
def pause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

if __name__ == "__main__":
    # set alpha values to test (correlation between inputs = 1 - alpha)
    alpha_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]

    for alpha in alpha_values:
        # set experiment parameters
        n_trials     = 200
        layer_1_size = 1

        # initialize lists for recording results
        ws                         = []
        RDD_values                 = []
        estimated_RDD_values       = [ [] for k in range(n_trials) ]
        all_max_drives             = [ [] for k in range(n_trials) ]
        all_Rs                     = [ [] for k in range(n_trials) ]
        all_c_belows               = [ [] for k in range(n_trials) ]
        all_c_aboves               = [ [] for k in range(n_trials) ]
        all_m_belows               = [ [] for k in range(n_trials) ]
        all_m_aboves               = [ [] for k in range(n_trials) ]
        layer_1_vs                 = [ [] for k in range(n_trials) ]
        layer_1_us                 = [ [] for k in range(n_trials) ]
        layer_1_spikes             = [ [] for k in range(n_trials) ]
        layer_1_inputs             = [ [] for k in range(n_trials) ]
        layer_1_feedbacks          = [ [] for k in range(n_trials) ]
        layer_2_vs                 = [ [] for k in range(n_trials) ]
        layer_2_spikes             = [ [] for k in range(n_trials) ]
        layer_2_inputs             = [ [] for k in range(n_trials) ]
        all_RDD_times              = [ [] for k in range(n_trials) ]
        final_estimated_RDD_values = []

        # set feedforward weight values that will be tested
        ws = np.random.uniform(-1000, 1000, size=(n_trials, layer_1_size))

        # create feedback weights
        y = np.ones((layer_1_size, 1))

        if updating_plot:
            # create live updating plot of weights vs. estimated RDD values
            plt.ion()
            fig, ax = plt.subplots()
            sc = ax.scatter([], [], c='g', alpha=0.5)
            sc2 = ax.scatter([], [], c='y', alpha=0.5)
            plt.axvline(x=0, color='k', linestyle='--')
            plt.axhline(y=0, color='k', linestyle='--')
            plt.xlabel("Feedforward Weight")
            plt.ylabel("Estimated RDD Value")
            plt.xlim(-1000, 1000)
            plt.draw()

        for k in range(n_trials):
            print("Trial {}/{}.".format(k+1, n_trials))

            # create 2 layers, 1 neuron each
            layer_1 = Layer(size=layer_1_size)
            layer_2 = FinalLayer(size=1)

            # set the weight from neuron 1 to neuron 2
            w = ws[k]

            # set total time of simulation (timesteps)
            total_time = int(100/dt)

            # create array of timesteps
            time_array = np.arange(total_time)

            # create inputs for layer 1 and 2 with a degree of correlation determined by alpha (alpha=1 means no correlation)
            input_1 = 100*(np.random.normal(0, 1, size=(layer_1_size, total_time)) + 0.5)
            input_2 = 100*(np.random.normal(0, 1, size=total_time) + 0.5)
            input_3 = 100*(np.random.normal(0, 1, size=total_time) + 0.5)

            input_1 = np.sqrt(alpha)*input_1 + np.sqrt(1 - alpha)*input_3
            input_2 = np.sqrt(alpha)*input_2 + np.sqrt(1 - alpha)*input_3

            # create some more recording arrays
            below_max_drives        = [ [] for j in range(layer_1_size) ]
            above_max_drives        = [ [] for j in range(layer_1_size) ]
            below_Rs                = [ [] for j in range(layer_1_size) ]
            above_Rs                = [ [] for j in range(layer_1_size) ]
            window_start_times      = [ [] for j in range(layer_1_size) ]

            print("Running simulation...")

            for t in range(total_time):
                # create weighted, convolved feedforward and feedback inputs to each layer
                layer_1_input    = input_1[:, t][:, np.newaxis]
                layer_2_input    = np.dot(w, np.dot(layer_1.spike_hist, kappas)) + input_2[t]
                layer_1_feedback = np.dot(y, np.dot(layer_2.spike_hist, kappas))

                # update layer 1 and 2 activities
                layer_1.update(layer_1_input, feedback=layer_1_feedback, time=t)
                layer_2.update(layer_2_input)

                # record variables
                estimated_RDD_value = (layer_1.RDD_params[:, 2]*v_threshold + layer_1.RDD_params[:, 0]) - (layer_1.RDD_params[:, 3]*v_threshold + layer_1.RDD_params[:, 1])
                estimated_RDD_values[k].append(estimated_RDD_value)
                all_c_aboves[k].append(layer_1.RDD_params[:, 0])
                all_m_aboves[k].append(layer_1.RDD_params[:, 2])
                all_c_belows[k].append(layer_1.RDD_params[:, 1])
                all_m_belows[k].append(layer_1.RDD_params[:, 3])
                layer_1_vs[k].append(layer_1.v)
                layer_1_us[k].append(layer_1.u)
                layer_1_spikes[k].append(layer_1.fired)
                layer_1_inputs[k].append(layer_1_input)
                layer_1_feedbacks[k].append(layer_1_feedback)
                layer_2_vs[k].append(layer_2.v)
                layer_2_spikes[k].append(layer_2.fired)
                layer_2_inputs[k].append(layer_2_input)
                estimated_RDD_values[k].append(estimated_RDD_value)
                estimated_RDD_values[k].append(estimated_RDD_value)
                estimated_RDD_values[k].append(estimated_RDD_value)

                for j in layer_1.neurons_updated_below:
                    below_max_drives[j].append(layer_1.max_u)
                    below_Rs[j].append(layer_1.R)

                    all_max_drives[k].append(layer_1.max_u)
                    all_Rs[k].append(layer_1.R)
                    all_RDD_times[k].append(t)
                for j in layer_1.neurons_updated_above:
                    above_max_drives[j].append(layer_1.max_u)
                    above_Rs[j].append(layer_1.R)

                    all_max_drives[k].append(layer_1.max_u)
                    all_Rs[k].append(layer_1.R)
                    all_RDD_times[k].append(t)

                # reset layer 1 RDD variables
                layer_1.reset_RDD_variables()

                if updating_plot:
                    # update the live plot
                    if t % 1000 == 0:
                        sc2.set_offsets(np.c_[ws[k], estimated_RDD_value])
                        if k == 0:
                            scale = np.amax(np.abs(estimated_RDD_value))
                        else:
                            scale = np.maximum(np.amax(np.abs(final_estimated_RDD_values)), np.amax(np.abs(estimated_RDD_value)))
                        plt.ylim(-scale, scale)
                        fig.canvas.draw_idle()
                        plt.title('Trial {}. Time: {} s.'.format(k+1, int(t*dt)))
                        pause(0.00001)

            print("Below data points: {}".format(len(below_max_drives[0])))
            print("Above data points: {}".format(len(above_max_drives[0])))

            # compute a 'true' RDD value for each unit in layer 1 using least squares method
            RDD_value = np.zeros(layer_1_size)
            for j in range(layer_1_size):
                try:
                    # perform two linear regressions on samples below threshold and above threshold
                    A = np.vstack([np.array(below_max_drives[j]), np.ones(len(np.array(below_max_drives[j])))]).T
                    m_below, c_below = np.linalg.lstsq(A, np.array(below_Rs[j]).T)[0]

                    A = np.vstack([np.array(above_max_drives[j]), np.ones(len(np.array(above_max_drives[j])))]).T
                    m_above, c_above = np.linalg.lstsq(A, np.array(above_Rs[j]).T)[0]
                except:
                    m_above = 0
                    c_above = 0
                    m_below = 0
                    c_below = 0

                RDD_value[j] = (m_above*v_threshold + c_above) - (m_below*v_threshold + c_below)

            # compute the final estimated RDD value for each unit in layer 1
            m_above_est = layer_1.RDD_params[:, 2]
            c_above_est = layer_1.RDD_params[:, 0]
            m_below_est = layer_1.RDD_params[:, 3]
            c_below_est = layer_1.RDD_params[:, 1]
            final_estimated_RDD_value = (layer_1.RDD_params[:, 2]*v_threshold + layer_1.RDD_params[:, 0]) - (layer_1.RDD_params[:, 3]*v_threshold + layer_1.RDD_params[:, 1])

            # create an array that is 1 when an RDD window was occurring, and 0 otherwise
            RDD_window_array = np.zeros(time_array.shape)
            for t in all_RDD_times[k]:
                RDD_window_array[t-int(RDD_window)+1:t+1] = 1
            
            print("Neuron 1 -> 2 weight: {}.".format(w))
            print("RDD value: {}.".format(RDD_value))
            print("Estimated RDD value: {}.".format(final_estimated_RDD_value))

            RDD_values.append(RDD_value)
            final_estimated_RDD_values.append(final_estimated_RDD_value)

            if updating_plot:
                # update the live plot
                sc.set_offsets(np.c_[ws[:k+1], final_estimated_RDD_values])
                sc2.set_offsets(np.c_[[], []])
                scale = np.amax(np.abs(final_estimated_RDD_values))
                plt.ylim(-scale, scale)
                fig.canvas.draw_idle()

                pause(0.0001)

        # save recorded variables
        np.save('alt_1_unit_lstsq_correlation_{}_ws.npy'.format(1-alpha), ws)
        np.save('alt_1_unit_lstsq_correlation_{}_final_estimated_RDD_values.npy'.format(1-alpha), final_estimated_RDD_values)
        np.save('alt_1_unit_lstsq_correlation_{}_estimated_RDD_values.npy'.format(1-alpha), estimated_RDD_values)
        np.save('alt_1_unit_lstsq_correlation_{}_max_drives.npy'.format(1-alpha), all_max_drives)
        np.save('alt_1_unit_lstsq_correlation_{}_Rs.npy'.format(1-alpha), all_Rs)
        np.save('alt_1_unit_lstsq_correlation_{}_all_c_belows.npy'.format(1-alpha), all_c_belows)
        np.save('alt_1_unit_lstsq_correlation_{}_all_m_belows.npy'.format(1-alpha), all_m_belows)
        np.save('alt_1_unit_lstsq_correlation_{}_all_c_aboves.npy'.format(1-alpha), all_c_aboves)
        np.save('alt_1_unit_lstsq_correlation_{}_all_m_aboves.npy'.format(1-alpha), all_m_aboves)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_1_vs.npy'.format(1-alpha), layer_1_vs)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_1_us.npy'.format(1-alpha), layer_1_us)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_1_spikes.npy'.format(1-alpha), layer_1_spikes)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_1_inputs.npy'.format(1-alpha), layer_1_inputs)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_1_feedbacks.npy'.format(1-alpha), layer_1_feedbacks)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_2_vs.npy'.format(1-alpha), layer_2_vs)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_2_spikes.npy'.format(1-alpha), layer_2_spikes)
        np.save('alt_1_unit_lstsq_correlation_{}_layer_2_inputs.npy'.format(1-alpha), layer_2_inputs)
        np.save('alt_1_unit_lstsq_correlation_{}_all_RDD_times.npy'.format(1-alpha), all_RDD_times)

        # plot results for all trials
        plt.clf()
        plt.scatter([ ws[i] for i in range(n_trials) ], [ value for value in RDD_values ], c='g', alpha=0.5, label='Least Squares Fit RDD Value')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel("Feedforward Weight")
        plt.ylabel("RDD Value")
        plt.legend()
        plt.savefig('alt_1_unit_lstsq_correlation_{}.png'.format(1-alpha))
        plt.savefig('alt_1_unit_lstsq_correlation_{}.svg'.format(1-alpha))

        plt.clf()
        plt.scatter([ ws[i] for i in range(n_trials) ], [ value for value in final_estimated_RDD_values ], c='b', alpha=0.5, label='Estimated RDD Value')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel("Feedforward Weight")
        plt.ylabel("RDD Value")
        plt.legend()
        plt.savefig('alt_1_unit_RDD_estimate_correlation_{}.png'.format(1-alpha))
        plt.savefig('alt_1_unit_RDD_estimate_correlation_{}.svg'.format(1-alpha))
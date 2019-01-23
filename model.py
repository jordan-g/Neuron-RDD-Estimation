import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
dt               = 0.001             # timestep (s)
v_reset          = 0                 # reset voltage
v_threshold      = 1                 # spike threshold
tau              = 0.01              # synaptic time constant (s) - used to calculate voltage
refractory_time  = np.ceil(0.01/dt)  # refractory time (timesteps)
u_window         = 1                 # input drive window used to determine near-threshold data points for RDD
tau_s            = 0.003             # synaptic time constant (s) - used to calculate input
tau_L            = 0.01              # leak time constant (s) - used to calculate input
mem              = int(20/dt)        # spike history memory length (timesteps)
RDD_window       = 0.05/dt           # RDD integration window length (timesteps)
RDD_init_window  = 0.2               # window around spike threshold that determines when an RDD integration window is initiated

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
		self.RDD_windows            = []                                      # list of RDD integration window initiation times

		# RDD estimation parameters
		self.RDD_params = np.zeros((self.size, 4)) # list of RDD estimation parameters - gamma, beta, alpha_l and alpha_r
		self.eta        = 0.05                     # learning rate

	def update(self, I, feedback=None, time=0):
		if feedback is not None:
			self.feedback = feedback

			# determine which neurons are just ending their RDD integration window
			self.RDD_window_ending_mask = self.RDD_time_left == 1

			# update maximum input drives for units in their RDD integration window
			self.max_u[np.logical_and(self.RDD_time_left > 0, self.u > self.max_u)] = self.u[np.logical_and(self.RDD_time_left > 0, self.u > self.max_u)]

			# update rewards for units in their RDD integration window
			self.R[self.RDD_time_left > 0] += self.feedback[self.RDD_time_left > 0]

		# update refractory period timesteps remaining for each neuron
		self.refractory_time_left[self.refractory_time_left > 0] -= 1

		if feedback is not None:
			# reset the input drive to match the voltage, for neurons that are not in an RDD integration window
			self.u[self.RDD_time_left == 0] = self.v[self.RDD_time_left == 0]
		else:
			self.v = self.u

		# calculate changes in voltages and input drives, and update both
		self.dv_dt    = -self.v/(tau) + I
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

		if feedback is not None:
			# reset the input drive to match the voltage, for neurons that are not in an RDD integration window
			self.u[self.RDD_time_left == 0] = self.v[self.RDD_time_left == 0]
		else:
			self.v = self.u

		if feedback is not None:
			# update RDD estimates (only neurons whose RDD integration window has ended will update their estimate)
			self.update_RDD_estimate(time=time)

		# decrement time left in RDD integration windows
		self.RDD_time_left[self.RDD_time_left > 0] -= 1

		# determine which neurons are starting a new RDD integration window
		if feedback is not None:
			self.new_RDD_window_mask = np.logical_and(np.abs(v_threshold - self.u) <= RDD_init_window, self.RDD_time_left == 0)

			self.RDD_time_left[self.new_RDD_window_mask] = RDD_window
		else:
			self.new_RDD_window_mask = np.zeros((self.size, 1)).astype(bool)

		if feedback is not None:
			# reset RDD variables for neurons whose RDD integration window has ended
			self.n_spikes[self.RDD_window_ending_mask] = 0
			self.R[self.RDD_window_ending_mask]        = 0
			self.max_u[self.RDD_window_ending_mask]    = 0

		# update number of spikes that have occurred during RDD integration windows
		self.n_spikes[np.logical_and(self.RDD_time_left > 0, self.fired)] += 1

		# update spike histories
		self.spike_hist = np.concatenate([self.spike_hist[:, 1:], self.fired.astype(int)], axis=1)

	def update_RDD_estimate(self, time=0):
		# figure out which neurons are at the end of their RDD integration window, and either just spiked or almost spiked
		just_spiked_mask   = np.logical_and(self.RDD_window_ending_mask, np.logical_and(np.abs(self.max_u - v_threshold) <= u_window, self.max_u >= v_threshold))[:, 0]
		almost_spiked_mask = np.logical_and(self.RDD_window_ending_mask, np.logical_and(np.abs(self.max_u - v_threshold) <= u_window, self.max_u < v_threshold))[:, 0]

		# create an array used to update RDD estimates
		a = np.ones((4, self.size))
		a[1, :] = (self.n_spikes >= 1).astype(float)
		a[2, :] = (self.n_spikes >= 1).astype(float)*(self.max_u - v_threshold)
		a[3, :] = (1 - (self.n_spikes >= 1).astype(float))*(self.max_u - v_threshold)

		# update RDD estimates for neurons that just spiked or almost spiked
		if np.sum(just_spiked_mask) > 0:
			self.RDD_params[just_spiked_mask]   -= np.array([ self.eta*(np.dot(self.RDD_params[i], a[:, i]) - self.R[i])*a[:, i] for i in np.where(just_spiked_mask)[0] ])

			self.RDD_windows.append(time)
		elif np.sum(almost_spiked_mask) > 0:
			self.RDD_params[almost_spiked_mask] -= np.array([ self.eta*(np.dot(self.RDD_params[i], a[:, i]) - self.R[i])*a[:, i] for i in np.where(almost_spiked_mask)[0] ])

			self.RDD_windows.append(time)

if __name__ == "__main__":
	# set experiment parameters
	n_trials     = 100
	layer_1_size = 1

	# initialize lists for recording results
	ws                         = []
	RDD_values                 = []
	final_estimated_RDD_values = []

	# set weight values that will be tested
	ws = np.zeros((n_trials, layer_1_size))
	for i in range(layer_1_size):
		ws[:, i] = np.linspace(-1000, 1000, n_trials)

	for k in range(n_trials):
		print("Trial {}/{}.".format(k+1, n_trials))

		# create 2 layers, 1 neuron each
		layer_1 = Layer(size=layer_1_size)
		layer_2 = Layer(size=1)

		# set the weight from neuron 1 to neuron 2
		w = ws[k]

		# set total time of simulation (timesteps)
		total_time = int(60/dt)

		# create array of timesteps
		time_array = np.arange(total_time)

		# create inputs for layer 1 and 2 with a degree of correlation determined by alpha (1 means no correlation)
		alpha = 0.5

		input_1 = 200*(np.random.normal(0, 1, size=(layer_1_size, total_time)))
		input_2 = 200*(np.random.normal(0, 1, size=total_time))
		input_3 = 200*(np.random.normal(0, 1, size=total_time))

		input_1 = np.sqrt(alpha)*input_1 + np.sqrt(1 - alpha)*input_3
		input_2 = np.sqrt(alpha)*input_2 + np.sqrt(1 - alpha)*input_3

		# initialize recording arrays
		voltages_1           = np.zeros((layer_1_size, total_time))
		voltages_2           = np.zeros((1, total_time))
		drives_1             = np.zeros((layer_1_size, total_time))
		drives_2             = np.zeros((1, total_time))
		feedbacks            = np.zeros((1, total_time))
		estimated_RDD_values = np.zeros((1, total_time))

		below_us                = [ [] for j in range(layer_1_size) ]
		above_us                = [ [] for j in range(layer_1_size) ]
		below_Rs                = [ [] for j in range(layer_1_size) ]
		above_Rs                = [ [] for j in range(layer_1_size) ]
		window_start_times      = [ [] for j in range(layer_1_size) ]
		kept_window_start_times = [ [] for j in range(layer_1_size) ]

		print("Running simulation...")

		for t in range(total_time):
			# update layer 1 and 2 activities
			layer_1.update(input_1[:, t][:, np.newaxis], feedback=np.dot(layer_2.spike_hist, kappas), time=t)
			layer_2.update(np.dot(w, np.dot(layer_1.spike_hist, kappas)) + input_2[t])

			# record voltages and input drives
			voltages_1[:, t]           = layer_1.v[:, 0]
			voltages_2[:, t]           = layer_2.v[:, 0]
			drives_1[:, t]             = layer_1.u[:, 0]
			drives_2[:, t]             = layer_2.u[:, 0]
			feedbacks[:, t]            = layer_1.feedback[:, 0]
			estimated_RDD_values[:, t] = layer_1.RDD_params[0, 1]

		# compute RDD points after-the-fact (for troubleshooting purposes)
		for t in range(total_time):
			for j in range(layer_1_size):
				if len(window_start_times[j]) == 0 or window_start_times[j][-1] + RDD_window <= t <= total_time - RDD_window:
					if np.abs(v_threshold - drives_1[j, t]) <= RDD_init_window:
						# calculate maximum input drive
						max_drive = np.amax(drives_1[j, t:t+int(RDD_window)])

						window_start_times[j].append(t)

						if np.abs(v_threshold - max_drive) <= u_window:
							# compute reward
							R = np.sum(feedbacks[:, t+1:t+int(RDD_window)+1])

							if max_drive < v_threshold:
								below_us[j].append(max_drive)
								below_Rs[j].append(R)
								kept_window_start_times[j].append(t)
							else:
								above_us[j].append(max_drive)
								above_Rs[j].append(R)
								kept_window_start_times[j].append(t)

		print(len(layer_1.RDD_windows))
		print(len(kept_window_start_times[0]))

		print("Below data points: {}".format(len(below_us[0])))
		print("Above data points: {}".format(len(above_us[0])))

		below_us     = np.array(below_us)
		above_us     = np.array(above_us)
		below_Rs     = np.array(below_Rs)
		above_Rs     = np.array(above_Rs)

		try:
			# perform two linear regressions on samples below threshold and above threshold
			A = np.vstack([below_us[0], np.ones(len(below_us[0]))]).T
			m_below, c_below = np.linalg.lstsq(A, below_Rs[0].T)[0]

			A = np.vstack([above_us[0], np.ones(len(above_us[0]))]).T
			m_above, c_above = np.linalg.lstsq(A, above_Rs[0].T)[0]
		except:
			m_above = 0
			c_above = 0
			m_below = 0
			c_below = 0

		# calculate the true RDD value (beta), and record the one estimated by layer 1 neurons
		RDD_value           = (m_above*v_threshold + c_above) - (m_below*v_threshold + c_below)
		# RDD_value           = np.mean(above_Rs, axis=-1) - np.mean(below_Rs, axis=-1)

		final_estimated_RDD_value = layer_1.RDD_params[0, 1]

		if k == 0:
			plt.plot(estimated_RDD_values[0, :])
			plt.plot(np.arange(total_time), RDD_value*np.ones(total_time))
			plt.show()

		# create an array that is 1 when an RDD window was occurring, and 0 otherwise
		RDD_window_array = np.zeros(time_array.shape)
		for t in kept_window_start_times[0]:
			RDD_window_array[t:t+int(RDD_window)] = 1

		# plot results for the first trial
		if k == 0:
			plt.subplot(211)

			# plt.plot(input_1[0], 'g')
			# plt.plot(input_2, 'purple')
			plt.plot(voltages_1[0], 'r', alpha=0.5)
			plt.plot(feedbacks[0], 'b', alpha=0.5)
			plt.plot(drives_1[0], 'r', linestyle='--', alpha=0.5)
			# plt.plot(drives_2[0], 'b', linestyle='--', alpha=0.5)
			plt.plot(time_array, v_threshold*np.ones(total_time), 'k', linestyle='--')
			plt.xlim(0, total_time)

			plt.gca().fill_between(np.arange(total_time), 0, 1, where=time_array*RDD_window_array > 0, facecolor='green', alpha=0.3)

			plt.xlabel("Time (timesteps)")

			plt.subplot(212)

			plt.scatter(below_us, below_Rs, c='b', alpha=0.5)
			plt.scatter(above_us, above_Rs, c='r', alpha=0.5)

			x = np.linspace(v_threshold - u_window, v_threshold, 100)
			plt.plot(x, m_below*x + c_below, 'b', linestyle='--')

			x = np.linspace(v_threshold, v_threshold + u_window, 100)
			plt.plot(x, m_above*x + c_above, 'r', linestyle='--')

			x = np.linspace(v_threshold - u_window, v_threshold, 100)
			plt.plot(x, np.mean(below_Rs)*np.ones(100), 'b', linestyle='--', alpha=0.5)

			x = np.linspace(v_threshold, v_threshold + u_window, 100)
			plt.plot(x, np.mean(above_Rs)*np.ones(100), 'r', linestyle='--', alpha=0.5)

			plt.axvline(x=v_threshold, color='k', linestyle='--')
			plt.xlim(v_threshold - u_window, v_threshold + u_window)

			plt.xlabel("Neuron 1 maximum input drive")
			plt.ylabel("Neuron 2 spiking activity")

			plt.show()
		
		print("Neuron 1 -> 2 weight: {}.".format(w))
		print("RDD value: {}.".format(RDD_value))
		print("Estimated RDD value: {}.".format(final_estimated_RDD_value))
		RDD_values.append(RDD_value)
		final_estimated_RDD_values.append(final_estimated_RDD_value)

	# plot results for all trials
	plt.scatter([ ws[i, 0] for i in range(n_trials) ], [ value for value in RDD_values ], c='g', alpha=0.5, label='RDD Value')
	plt.scatter([ ws[i, 0] for i in range(n_trials) ], [ value for value in final_estimated_RDD_values ], c='b', alpha=0.5, label='Estimated RDD Value')
	plt.axvline(x=0, color='k', linestyle='--')
	plt.axhline(y=0, color='k', linestyle='--')
	plt.xlabel("Neuron 1 -> 2 Weight")
	plt.ylabel("RDD Value")
	plt.legend()
	plt.show()
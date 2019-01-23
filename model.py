import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
dt              = 0.001             # timestep (s)
v_reset         = 0                 # reset voltage
v_threshold     = 1                 # spike threshold
tau             = 0.01              # synaptic time constant (s)
refractory_time = np.ceil(0.01/dt)  # refractory time (timesteps)
u_window        = 1                 # RDD input drive window
tau_s           = 0.003             # synaptic time constant (s)
tau_L           = 0.01              # leak time constant (s)
mem             = int(20/dt)        # spike history memory length (timesteps)
reset_time      = 0.05/dt
u_window_2      = 0.2

# KERNEL FUNCTION
def kappa(x):
    return (np.exp(-x/(tau_L/dt)) - np.exp(-x/(tau_s/dt)))/((tau_L/dt) - (tau_s/dt))

def get_kappas(n=mem):
    return np.array([kappa(i+1) for i in range(n)])

kappas = np.flipud(get_kappas(mem))[:, np.newaxis] # initialize kappas array

class Layer():
	def __init__(self, size):
		self.size                 = size
		self.u                    = v_reset*np.ones((self.size, 1))
		self.max_u                = np.zeros((self.size, 1))
		self.v                    = v_reset*np.ones((self.size, 1))
		self.dv_dt                = np.zeros((self.size, 1))
		self.fired                = np.zeros((self.size, 1)).astype(bool)
		self.refractory_time_left = np.zeros((self.size, 1))
		self.reset_time_left      = np.zeros((self.size, 1))
		self.spike_hist           = np.zeros((self.size, mem), dtype=np.int8)
		self.feedback             = np.zeros((self.size, 1))
		self.y                    = np.random.uniform(0, 1, size=(self.size, 1))
		self.R                    = np.zeros((self.size, 1))
		self.n_spikes             = np.zeros((self.size, 1))
		self.window_times         = []

		# RDD estimation parameters
		self.RDD_params = np.zeros((self.size, 4))
		self.eta        = 0.01

	def update(self, I, feedback=None, time=0):
		if feedback is not None:
			# determine which neurons are just ending their window
			self.window_ending_mask = self.reset_time_left == 1

			# if np.sum(self.window_ending_mask) > 0 and feedback is not None:
			# 	print("- {}".format(time))
				# print(time)

			self.max_u[np.logical_and(self.reset_time_left > 0, self.u > self.max_u)] = self.u[np.logical_and(self.reset_time_left > 0, self.u > self.max_u)]

			self.feedback = feedback

			self.R[self.reset_time_left > 0] += self.feedback[self.reset_time_left > 0]

			self.reset_time_left[self.reset_time_left > 0] -= 1

		# update refractory period timesteps remaining for each neuron
		self.refractory_time_left[self.refractory_time_left > 0] -= 1

		# calculate change in voltage and input drive, and update both
		self.dv_dt    = -self.v/(tau) + I
		self.u       += dt*self.dv_dt
		self.v       += dt*self.dv_dt

		# determine which neurons are in a refractory period
		refractory_mask = self.refractory_time_left > 0

		# determine which neurons are above spiking threshold
		threshold_mask  = self.v >= v_threshold

		if feedback is not None:
			self.update_RDD_estimate(time=time)
			
			self.u[self.window_ending_mask] = self.v[self.window_ending_mask]

		# determine which neurons are starting a new window
		self.new_window_mask = np.logical_and(np.abs(v_threshold - self.u) <= u_window_2, self.reset_time_left == 0)

		if feedback is None:
			self.new_window_mask *= False

		if feedback is None:
			self.u = self.v

		# if np.sum(self.new_window_mask) > 0 and feedback is not None:
		# 	print("New window at times {}".format(time), end="")
		# 	# print(time)
		# 	# print(t, self.u[0, 0])
		# 	self.window_times.append(time)

		# neurons above threshold will spike
		self.fired[threshold_mask] = True

		# make sure neurons do not spike when in the refractory period
		self.fired[refractory_mask] = False

		# reset voltages of neurons that spiked
		self.v[self.fired] = v_reset

		if feedback is not None:
			self.u[self.reset_time_left == 0] = self.v[self.reset_time_left == 0]

		# reset input drive if a neuron has spiked and the refractory period is ending
		# self.u[self.new_window_mask] = self.v[self.new_window_mask]

			self.n_spikes[self.window_ending_mask] = 0
			self.R[self.window_ending_mask] = 0
			self.max_u[self.window_ending_mask] = 0

			self.reset_time_left[self.new_window_mask] = reset_time

		# update refractory period timesteps remaining for each neuron
		self.refractory_time_left[self.fired] = refractory_time

		self.n_spikes[np.logical_and(self.reset_time_left > 0, self.fired)] += 1

		# if np.sum(self.reset_time_left) == reset_time+1 and feedback is not None:
		# 	print("Reset at time {}".format(time))

		# update spike history
		self.spike_hist = np.concatenate([self.spike_hist[:, 1:], self.fired.astype(int)], axis=1)

	def update_RDD_estimate(self, time=0):
		# if np.sum(self.window_ending_mask) > 0:
		# 	print(time, self.max_u[0, 0])

		# figure out which neurons just spiked or almost spiked
		just_spiked_mask   = np.logical_and(self.window_ending_mask, np.logical_and(np.abs(self.max_u - v_threshold) <= u_window, self.max_u >= v_threshold))[:, 0]
		almost_spiked_mask = np.logical_and(self.window_ending_mask, np.logical_and(np.abs(self.max_u - v_threshold) <= u_window, self.max_u < v_threshold))[:, 0]

		# create array used to update RDD estimates
		a = np.ones((4, self.size))
		a[1, :] = (self.n_spikes >= 1).astype(float)
		a[2, :] = (self.n_spikes >= 1).astype(float)*(self.max_u - v_threshold)
		a[3, :] = (1 - (self.n_spikes >= 1).astype(float))*(self.max_u - v_threshold)

		# print(a)

		# update RDD estimates for neurons that just spiked or almost spiked
		if np.sum(just_spiked_mask) > 0:
			self.RDD_params[just_spiked_mask]   -= np.array([ self.eta*(np.dot(self.RDD_params[i], a[:, i]) - self.R[i])*a[:, i] for i in np.where(just_spiked_mask)[0] ])
			# self.y[just_spiked_mask] -= np.array([ self.eta*(self.y[i] - self.R[i]) for i in np.where(just_spiked_mask)[0] ])
			
			# print(self.RDD_params[just_spiked_mask, 0])
			# print("just", time)
			# print(a)
			# print(self.n_spikes)
			# print("u", self.max_u[0, 0])
			# print("R", self.R[0, 0])

			# print(self.n_spikes[0, 0], time)
			# print(time, self.max_u[0, 0])
			# print(time, self.R[0, 0])

			self.window_times.append(time)

			# print(self.reset_time_left[0, 0], self.refractory_time_left[0, 0], )
		elif np.sum(almost_spiked_mask) > 0:
			self.RDD_params[almost_spiked_mask] -= np.array([ self.eta*(np.dot(self.RDD_params[i], a[:, i]) - self.R[i])*a[:, i] for i in np.where(almost_spiked_mask)[0] ])
			# self.y[almost_spiked_mask] -= np.array([ self.eta*(self.y[i] + self.R[i]) for i in np.where(almost_spiked_mask)[0] ])

			# print(self.RDD_params[almost_spiked_mask, 0])
			# print("almost", time)
			# print("u", self.max_u[0, 0])
			# print("R", self.R[0, 0])

			# print(time, self.max_u[0, 0])
			# print(time, self.R[0, 0])

			self.window_times.append(time)

			# print(self.reset_time_left[0, 0], self.refractory_time_left[0, 0], )

if __name__ == "__main__":
	n_trials     = 100
	layer_1_size = 1

	ws             = []
	vals           = []
	estimated_vals = []

	# ws = np.linspace(-500, 500, n_trials)

	for k in range(n_trials):
		print("Trial {}/{}.".format(k+1, n_trials))

		# create 2 layers, 1 neuron each
		layer_1 = Layer(size=layer_1_size)
		layer_2 = Layer(size=1)

		# set weight from neuron 1 to neuron 2 randomly
		w = 500*np.random.uniform(-5, 5, size=(1, layer_1_size))
		# w = 5000*np.ones((1, layer_1_size))

		# set total time of simulation (timesteps)
		total_time = int(60/dt)

		# create array of timesteps
		time_array = np.arange(total_time)

		# create independent inputs for layer 1 and 2
		input_1 = np.zeros((layer_1_size, total_time))
		for j in range(layer_1_size):
			input_1[j] = 200*(np.random.normal(0, 1, size=total_time))
			# input_1[j, 1000:2000] = 10

		input_2 = 200*(np.random.normal(0, 1, size=total_time))

		input_3 = 200*(np.random.normal(0, 1, size=total_time))

		alpha = 0.5
		input_1 = np.sqrt(alpha)*input_1 + np.sqrt(1 - alpha)*input_3
		input_2 = np.sqrt(alpha)*input_2 + np.sqrt(1 - alpha)*input_3
		# input_2 = 0*(np.random.normal(0, 1, size=total_time) + 0.5)

		# initialize recording arrays
		voltages_1   = np.zeros((layer_1_size, total_time))
		voltages_2   = np.zeros((1, total_time))
		drives_1     = np.zeros((layer_1_size, total_time))
		drives_2     = np.zeros((1, total_time))
		feedbacks    = np.zeros((1, total_time))

		below_us           = [ [] for j in range(layer_1_size) ]
		above_us           = [ [] for j in range(layer_1_size) ]
		below_Rs           = [ [] for j in range(layer_1_size) ]
		above_Rs           = [ [] for j in range(layer_1_size) ]
		window_start_times = [ [] for j in range(layer_1_size) ]
		kept_window_start_times = [ [] for j in range(layer_1_size) ]

		print("Running simulation...")

		est = []

		for t in range(total_time):
			# update layer 1 and 2 activities
			layer_1.update(input_1[:, t][:, np.newaxis], feedback=np.dot(layer_2.spike_hist, kappas), time=t)
			layer_2.update(np.dot(w, np.dot(layer_1.spike_hist, kappas)) + input_2[t])

			# if np.sum(layer_1.new_window_mask) == 1:
				# print(t)

			est.append(layer_1.RDD_params[0, 1])

			# record voltages and input drives
			voltages_1[:, t] = layer_1.v[:, 0]
			voltages_2[:, t] = layer_2.v[:, 0]
			drives_1[:, t]   = layer_1.u[:, 0]
			drives_2[:, t]   = layer_2.u[:, 0]
			feedbacks[:, t]  = layer_1.feedback[:, 0]


		RDD_params = np.zeros(3)

		for t in range(total_time):
			for j in range(layer_1_size):
				if len(window_start_times[j]) == 0 or window_start_times[j][-1] + reset_time <= t <= total_time - reset_time:
					if np.abs(v_threshold - drives_1[j, t]) <= u_window_2:
						max_drive = np.amax(drives_1[j, t:t+int(reset_time)])
						# print(t+1, t+int(reset_time)+1)
						window_start_times[j].append(t)

						# print(t+int(reset_time)+1)
						# print(t)
						# print(t, drives_1[j, t])

						if np.abs(v_threshold - max_drive) <= u_window:

							# print(t+int(reset_time)+1)
							# print(t)
							# print(feedbacks[:, t+1:t+int(reset_time)+2].shape)

							# mean_R    = np.mean(feedbacks[:, t+1:t+int(reset_time)])
							mean_R    = np.sum(feedbacks[:, t+1:t+int(reset_time)+1])

							# a = np.ones(3)
							# a[1] = (max_drive >= v_threshold).astype(float)*(max_drive - v_threshold)
							# a[2] = (1 - (max_drive >= v_threshold).astype(float))*(max_drive - v_threshold)

							if max_drive < v_threshold:
								# print(max_drive)
								# print(mean_R)
								# print(t+int(reset_time), max_drive)
								# print(t+int(reset_time), mean_R, max_drive)
								below_us[j].append(max_drive)
								below_Rs[j].append(mean_R)
								kept_window_start_times[j].append(t)

								# RDD_params -= layer_1.eta*(np.dot(RDD_params, a) + mean_R)*a
							else:
								# print(t+int(reset_time), max_drive)
								# print(max_drive)
								# print(t+int(reset_time), mean_R)
								above_us[j].append(max_drive)
								above_Rs[j].append(mean_R)
								kept_window_start_times[j].append(t)

								# RDD_params -= layer_1.eta*(np.dot(RDD_params, a) - mean_R)*a

							# print(RDD_params[0])

		print(len(layer_1.window_times))
		print(len(kept_window_start_times[0]))

		# print(layer_1.window_times)
		# print(window_start_times[0])

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

		# calculate true RDD value (beta), and record the one estimated by layer 1 neurons
		rdd_value           = (m_above*v_threshold + c_above) - (m_below*v_threshold + c_below)
		# rdd_value           = np.mean(above_Rs, axis=-1) - np.mean(below_Rs, axis=-1)
		estimated_rdd_value = layer_1.RDD_params[:, 1]

		if k == 0:
			plt.plot(est)
			plt.plot(np.arange(len(est)), rdd_value*np.ones(len(est)))
			plt.show()

		window_array = np.zeros(time_array.shape)

		for t in kept_window_start_times[0]:
			window_array[t:t+int(reset_time)] = 1

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

			plt.gca().fill_between(np.arange(total_time), 0, 1, where=time_array*window_array > 0, facecolor='green', alpha=0.3)

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
		print("RDD value: {}.".format(rdd_value))
		print("Estimated RDD value: {}.".format(estimated_rdd_value))

		ws.append(w)
		vals.append(rdd_value)
		estimated_vals.append(estimated_rdd_value)

	# plot results for all trials
	plt.scatter([ w[0, 0] for w in ws ], [ val for val in vals ], c='g', alpha=0.5, label='RDD Value')
	plt.scatter([ w[0, 0] for w in ws ], [ val[0] for val in estimated_vals ], c='b', alpha=0.5, label='Estimated RDD Value')
	plt.axvline(x=0, color='k', linestyle='--')
	plt.axhline(y=0, color='k', linestyle='--')
	plt.xlabel("Neuron 1 -> 2 Weight")
	plt.ylabel("RDD Value")
	plt.legend()
	plt.show()
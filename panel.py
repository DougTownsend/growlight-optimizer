import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import copy

class panel:
    class panel_translation:
        def __init__(self, x_offset=0, y_offset=0, rotation=0):
            valid_rotations = [0, 90, 180, 270]
            if rotation not in valid_rotations:
                raise Exception(f"Allowed rotations are {valid_rotations}")
            self.x_offset = x_offset
            self.y_offset = y_offset
            self.rotation = rotation

    def __init__(self, width, length, min_led_spacing, x_offset=0, y_offset=0, rotation=0):
        self.width = width
        self.length = length
        self.min_led_spacing = min_led_spacing
        self.leds = []
        self.fixed_leds = []
        self.x_vals = np.linspace(-(width/2.0), width/2.0, int(width/min_led_spacing)+1)[1:-1]
        self.y_vals = np.linspace(-(length/2.0), length/2.0, int(length/min_led_spacing)+1)[1:-1]
        self.n_rows = len(self.y_vals)
        self.n_cols = len(self.x_vals)
        self.translations = [self.panel_translation(x_offset, y_offset, rotation)]
        #t0 means the first copy of the panel (translations[0])
        self.t0_x_vals = np.copy(self.x_vals)
        self.t0_y_vals = np.copy(self.y_vals)
        if rotation in [90, 270]:
            tmp = self.t0_x_vals
            self.t0_x_vals = self.t0_y_vals
            self.t0_y_vals = tmp
        self.t0_x_vals += x_offset
        self.t0_y_vals += y_offset
        self.flux_x_lb = self.t0_x_vals[0]
        self.flux_x_ub = self.t0_x_vals[-1]
        self.flux_y_lb = self.t0_y_vals[0]
        self.flux_y_ub = self.t0_y_vals[-1]
        return
    
    def add_copy(self, x_offset, y_offset, rotation):
        self.translations.append(self.panel_translation(x_offset, y_offset, rotation))

    def load_panel_file(self, fname, fixed=False):
        f = open(fname, "r")
        for line in f.readlines():
            coord_str = line.split(",")
            coords = [float(coord_str[0]), float(coord_str[1])]
            if fixed:
                self.fixed_leds.append(coords)
            else:
                self.leds.append(coords)
        f.close()
    
    def save_panel_file(self, fname):
        f = open(fname, "w+")
        for led in self.leds:
            f.write(f"{led[0]},{led[1]}\n")
        f.close()

    def place_random_leds(self, count):
        if count > self.n_rows * self.n_cols:
            print(f"too many LEDs. Max {self.n_rows * self.n_cols}")
            return
        rng = np.random.default_rng()
        for i in range(count):
            while True:
                x = rng.choice(self.x_vals)
                y = rng.choice(self.y_vals)
                if ([x, y] not in self.leds) and ([x,y] not in self.fixed_leds):
                    self.leds.append([x, y])
                    break

class grow_area:
    def __init__(self, width, length, height, resolution=0.125):
        self.xmin = -width/2.0
        self.xmax = width/2.0
        self.ymin = -length/2.0
        self.ymax = length/2.0
        x_vals = np.linspace(self.xmin, self.xmax, int(width/resolution))
        y_vals = np.linspace(self.ymin, self.ymax, int(length/resolution))
        self.height = height
        self.resolution = resolution
        self.grid_xx, self.grid_yy = np.meshgrid(x_vals, y_vals, indexing='ij')
        self.photon_flux = np.zeros(self.grid_xx.shape)
        self.panels = []
        self.patches = []
        self.maxflux = 0

    def translate_point(self, point, x_offset, y_offset, rotation, reversed=False):
        #If reversed is true the opposite translation is done
        ret_point = [point[0],point[1]]
        if not reversed:
            if rotation == 90:
                tmp = -ret_point[0]
                ret_point[0] = ret_point[1]
                ret_point[1] = tmp
            elif rotation == 180:
                ret_point[0] *= -1
                ret_point[1] *= -1
            elif rotation == 270:
                tmp = -ret_point[1]
                ret_point[1] = ret_point[0]
                ret_point[0] = tmp
            
            ret_point[0] += x_offset
            ret_point[1] += y_offset

        if reversed:
            ret_point[0] -= x_offset
            ret_point[1] -= y_offset

            rotation = (-rotation) % 360
            if rotation == 90:
                tmp = -ret_point[0]
                ret_point[0] = ret_point[1]
                ret_point[1] = tmp
            elif rotation == 180:
                ret_point[0] *= -1
                ret_point[1] *= -1
            elif rotation == 270:
                tmp = -ret_point[1]
                ret_point[1] = ret_point[0]
                ret_point[0] = tmp

        return ret_point
    
    def draw(self, fname=None):
        f, axs = plt.subplots(2,1, figsize=(20,12),  tight_layout=True, height_ratios=[10,1]) 
        axs[0].set_aspect('equal')
        axs[0].set_xlim(self.xmin-self.resolution, self.xmax+self.resolution)
        axs[0].set_ylim(self.ymin-self.resolution, self.ymax+self.resolution)
        im = axs[0].imshow(self.photon_flux, animated=True, origin='lower', extent=(self.xmin-(self.resolution/2), self.xmax+(self.resolution/2), self.ymin-(self.resolution/2), self.ymax+(self.resolution/2)))
        scatter = axs[0].scatter([], [], color='red')
        scatter_fixed = axs[0].scatter([], [], color='white')
        im.set_clim((np.min(self.photon_flux), np.max(self.photon_flux)))
        im.set_array(np.transpose(self.photon_flux))
        tmp = []
        tmp_fixed = []
        for p in self.panels:
            for t in p.translations:
                for led in p.leds:
                    tmp.append(self.translate_point(led, t.x_offset, t.y_offset, t.rotation))
                for led in p.fixed_leds:
                    tmp_fixed.append(self.translate_point(led, t.x_offset, t.y_offset, t.rotation))
        scatter.set_offsets(tmp)
        if len(tmp_fixed) > 0:
            scatter_fixed.set_offsets(tmp_fixed)
        for patch in self.patches:
            axs[0].add_patch(copy(patch))
        
        flux_distribution = np.copy(self.photon_flux)
        flux_distribution = flux_distribution.flatten()
        if np.max(flux_distribution) > self.maxflux:
            self.maxflux = np.max(flux_distribution)
        
        axs[1].hist(flux_distribution, bins=100, range=(0,self.maxflux))

        if fname != None:
            f.savefig(fname)
        plt.close(f)

    def add_panel(self, p):
        self.panels.append(p)
        for translation in p.translations:
            w = p.width
            l = p.length
            if translation.rotation in [90, 270]:
                tmp = w
                w = l
                l = tmp
            x = translation.x_offset
            y = translation.y_offset
            rect = patches.Rectangle((x-(w/2.0), y-(l/2.0)), w, l, linewidth=1, edgecolor="black", facecolor="none")
            self.patches.append(rect)
    
    def calc_relative_flux_density(self, x_offset, y_offset, height, view_angle=120):
        flux_pixel_area = self.resolution**2
        if view_angle == 120:
            return(flux_pixel_area * (((height**2) / (((x_offset**2)+(y_offset**2)+(height**2))**(2)))/np.pi))
        else:
            angle_rad = np.radians(view_angle)
            m = (-1 * np.log(2))/(np.log(np.cos(angle_rad/2)))
            return flux_pixel_area * (((height**(m+1.0)) / ((x_offset**2) + (y_offset**2) + (height**2))**((m+3.0)/2.0)) / (np.pi / (((m+3)/2)-1)))
    
    def clear_flux(self):
        self.photon_flux *= 0

    def add_led_flux(self, led, view_angle=120):
        tmp = self.calc_relative_flux_density(self.grid_xx - led[0], self.grid_yy - led[1], self.height, view_angle=view_angle)
        self.photon_flux = self.photon_flux + tmp

    def subtract_led_flux(self, led):
        tmp = self.calc_relative_flux_density(self.grid_xx - led[0], self.grid_yy - led[1], self.height)
        self.photon_flux = self.photon_flux - tmp

    def init_photon_flux(self, view_angle=120):
        self.clear_flux()
        for p in self.panels:
            for t in p.translations:
                for led in p.leds:
                    tmp_led = self.translate_point(led, t.x_offset, t.y_offset, t.rotation)
                    self.add_led_flux(tmp_led, view_angle=view_angle)

class flux_optimizer:
    def __init__(self, grow, view_angle=120):
        self.g = grow
        self.g.init_photon_flux(view_angle=view_angle)
        self.framenum = 0
        self.min_variance = float("inf")
        self.masks = []

        #Create masks for each panel's first copy
        #   Set all locations within the panel to 0
        #   Set all locations outside the panel to infinity
        for p_idx in range(len(self.g.panels)):
            tmp_mask = np.zeros(self.g.photon_flux.shape)
            for i in range(len(tmp_mask)):
                for j in range(len(tmp_mask[i])):
                    point = [self.g.grid_xx[i][j], self.g.grid_yy[i][j]]
                    tmp_mask[i][j] += float('inf')
                    if (point[0] >= self.g.panels[p_idx].flux_x_lb and point[0] <= self.g.panels[p_idx].flux_x_ub) and (point[1] >= self.g.panels[p_idx].flux_y_lb and point[1] <= self.g.panels[p_idx].flux_y_ub):
                            tmp_mask[i][j] = 0
            self.masks.append(tmp_mask)
        
    def optimizer_step(self, view_angle=120):
        for p_idx in range(len(self.g.panels)):
            #Randomize order of LEDs. Probably unnecessary
            idxs = list(range(len(self.g.panels[p_idx].leds)))
            rng = np.random.default_rng()
            rng.shuffle(idxs)

            for i in range(len(self.g.panels[p_idx].leds)):
                #Check that the LED index is still valid
                if i >= len(self.g.panels[p_idx].leds):
                    break

                #Get the LED's location on the first copy of the current panel and remove its flux
                led = self.g.panels[p_idx].leds[i]
                led = self.g.translate_point(led, self.g.panels[p_idx].translations[0].x_offset, self.g.panels[p_idx].translations[0].y_offset, self.g.panels[p_idx].translations[0].rotation)
                self.g.subtract_led_flux(led)

                #Creat masked flux arrays for each unique panel (set all flux values outside the bounds of the panel to infinity)
                tmp_flux = []
                for j in range(len(self.g.panels)):
                    tmp_flux.append(np.copy(self.g.photon_flux))
                    tmp_flux[j] += self.masks[j]

                #Variables to store the best new loaction for the LED throughout the search
                new_led = [0,0]
                new_led_panel = 0
                min_flux = float("inf")

                #Search the masked flux arrays for the valid location with the lowest flux
                for j in range(len(self.g.panels)):
                    sorted_flux = np.sort(tmp_flux[j].flatten())
                    for minval in sorted_flux:
                        if minval > min_flux:
                            break
                        
                        #Find global coordinates (with grow_area scale) where flux == minval
                        min_idx = np.argwhere(tmp_flux[j] == minval)[0]
                        min_coords = [self.g.grid_xx[min_idx[0]][min_idx[1]], self.g.grid_yy[min_idx[0]][min_idx[1]]]

                        #Find closest global coordinates (with panel scale) to the grow area location
                        #This is needed since the panel and grow area can have different resolutions and will not always line up
                        min_difference = float('inf')
                        new_x_val = 0
                        new_y_val = 0
                        for x_val in self.g.panels[j].t0_x_vals:
                            absolute_difference = abs(x_val - min_coords[0])
                            if absolute_difference < min_difference:
                                min_difference = absolute_difference
                                new_x_val = x_val
                        min_difference = float('inf')
                        for y_val in self.g.panels[j].t0_y_vals:
                            absolute_difference = abs(y_val - min_coords[1])
                            if absolute_difference < min_difference:
                                min_difference = absolute_difference
                                new_y_val = y_val

                        #Convert global coordinates to panel's local coordinates
                        tmp_led = self.g.translate_point([new_x_val, new_y_val], self.g.panels[j].translations[0].x_offset, self.g.panels[j].translations[0].y_offset, self.g.panels[j].translations[0].rotation, reversed=True)
                        #Only save the new LED if there is not already an LED there
                        if ((tmp_led not in self.g.panels[j].leds) and (tmp_led not in self.g.panels[j].fixed_leds)) or ((p_idx == j) and (tmp_led == self.g.panels[j].leds[i])):
                            led = [new_x_val, new_y_val]
                            new_led = self.g.translate_point(led, self.g.panels[j].translations[0].x_offset, self.g.panels[j].translations[0].y_offset, self.g.panels[j].translations[0].rotation, reversed=True)
                            new_led_panel = j
                            min_flux = minval
                            break
                
                #Add the flux for the first copy of the panel here since the old LED's flux for that copy has already been removed
                self.g.add_led_flux(led)

                #Remove the LED flux for the old location and add for the new location on all other copies of the panel
                #Add the local panel coordinates of the new LED to the panel
                save_frame = False
                if p_idx == new_led_panel:
                    for t in self.g.panels[p_idx].translations[1:]:
                        led = self.g.translate_point(self.g.panels[p_idx].leds[i], t.x_offset, t.y_offset, t.rotation)
                        self.g.subtract_led_flux(led)
                        led = self.g.translate_point(new_led, t.x_offset, t.y_offset, t.rotation)
                        self.g.add_led_flux(led)
                    if self.g.panels[p_idx].leds[i] != new_led[:]:
                        save_frame = True 
                        self.g.panels[p_idx].leds[i] = new_led[:]
                else:
                    for t in self.g.panels[p_idx].translations[1:]:
                        led = self.g.translate_point(self.g.panels[p_idx].leds[i], t.x_offset, t.y_offset, t.rotation)
                        self.g.subtract_led_flux(led)
                    self.g.panels[p_idx].leds.pop(i)
                    self.g.panels[new_led_panel].leds.append(new_led[:])
                    for t in self.g.panels[new_led_panel].translations[1:]:
                        led = self.g.translate_point(new_led, t.x_offset, t.y_offset, t.rotation)
                        self.g.add_led_flux(led)

                #If the variance is the lowest found, save the LED panels and the grow area image
                variance = np.var(self.g.photon_flux)
                if p_idx == new_led_panel:
                    if new_led != self.g.panels[p_idx].leds[i]:
                        save_frame = True
                else:
                    save_frame = True
                if save_frame:
                    fname = f"./frames/{str(self.framenum).zfill(8)}.png"
                    self.g.draw(fname=fname)
                    self.framenum += 1
                    if variance < self.min_variance:
                        self.min_variance = variance
                        self.g.draw(fname="optimized_panel.png")
                        for j in range(len(self.g.panels)):
                            self.g.panels[j].save_panel_file(f"optimized_panel{j}.csv")

                


"""
TODO:
add the loop to the optimizer class instead of doing it in main
add animation to optimizer, clean up frames after
"""

    
if __name__ == "__main__":
    panels = []
    view_angle = 120

    '''
    view_angle = 130

    leds_per_panel = 5

    panels.append(panel(5, 5, .25, x_offset=-7.875, y_offset=7.875, rotation=0))
    panels[0].load_panel_file(f'./20x20/400led_4in/optimized_panel0.csv', fixed=True)
    panels[0].place_random_leds(leds_per_panel)
    panels[0].add_copy(7.875, 7.875, 90)
    panels[0].add_copy(7.875, -7.875, 180)
    panels[0].add_copy(-7.875, -7.875, 270)
    
    panels.append(panel(5, 5, .25, x_offset=-2.625, y_offset=7.875, rotation=0))
    panels[1].load_panel_file(f'./20x20/400led_4in/optimized_panel1.csv', fixed=True)
    panels[1].place_random_leds(leds_per_panel)
    panels[1].add_copy(7.872, 2.625, 90)
    panels[1].add_copy(2.625, -7.875, 180)
    panels[1].add_copy(-7.875, -2.625, 270)

    panels.append(panel(5, 5, .25, x_offset=-7.875, y_offset=2.625, rotation=0))
    panels[2].load_panel_file(f'./20x20/400led_4in/optimized_panel2.csv', fixed=True)
    panels[2].place_random_leds(leds_per_panel)
    panels[2].add_copy(2.625, 7.875, 90)
    panels[2].add_copy(7.875, -2.625, 180)
    panels[2].add_copy(-2.625, -7.875, 270)
    
    panels.append(panel(5, 5, .25, x_offset=-2.625, y_offset=2.625, rotation=0))
    panels[3].load_panel_file(f'./20x20/400led_4in/optimized_panel3.csv', fixed=True)
    panels[3].place_random_leds(leds_per_panel)
    panels[3].add_copy(2.625, 2.625, 90)
    panels[3].add_copy(2.625, -2.625, 180)
    panels[3].add_copy(-2.625, -2.625, 270)

    '''
    leds_per_panel = 25

    panels.append(panel(5, 5, .25, x_offset=-7.875, y_offset=7.875, rotation=0))
    panels[0].place_random_leds(leds_per_panel)
    panels[0].add_copy(7.875, 7.875, 90)
    panels[0].add_copy(7.875, -7.875, 180)
    panels[0].add_copy(-7.875, -7.875, 270)
    
    panels.append(panel(5, 5, .25, x_offset=-2.625, y_offset=7.875, rotation=0))
    panels[1].place_random_leds(leds_per_panel)
    panels[1].add_copy(7.872, 2.625, 90)
    panels[1].add_copy(2.625, -7.875, 180)
    panels[1].add_copy(-7.875, -2.625, 270)

    panels.append(panel(5, 5, .25, x_offset=-7.875, y_offset=2.625, rotation=0))
    panels[2].place_random_leds(leds_per_panel)
    panels[2].add_copy(2.625, 7.875, 90)
    panels[2].add_copy(7.875, -2.625, 180)
    panels[2].add_copy(-2.625, -7.875, 270)
    
    panels.append(panel(5, 5, .25, x_offset=-2.625, y_offset=2.625, rotation=0))
    panels[3].place_random_leds(leds_per_panel)
    panels[3].add_copy(2.625, 2.625, 90)
    panels[3].add_copy(2.625, -2.625, 180)
    panels[3].add_copy(-2.625, -2.625, 270)
    '''


    
    leds_per_panel = 19

    panels.append(panel(4.875, 4.875, .25, x_offset=-7.5, y_offset=7.5, rotation=0))
    panels[0].place_random_leds(leds_per_panel)
    panels[0].add_copy(7.5, 7.5, 0)
    panels[0].add_copy(7.5, -7.5, 180)
    panels[0].add_copy(-7.5, -7.5, 180)
    
    panels.append(panel(4.875, 4.875, .25, x_offset=-2.5, y_offset=7.5, rotation=0))
    panels[1].place_random_leds(leds_per_panel)
    panels[1].add_copy(2.5, 7.5, 0)
    panels[1].add_copy(2.5, -7.5, 180)
    panels[1].add_copy(-2.5, -7.5, 180)

    panels.append(panel(4.875, 4.875, .25, x_offset=-7.5, y_offset=2.5, rotation=0))
    panels[2].place_random_leds(leds_per_panel)
    panels[2].add_copy(7.5, 2.5, 180)
    panels[2].add_copy(7.5, -2.5, 180)
    panels[2].add_copy(-7.5, -2.5, 0)
    
    panels.append(panel(4.875, 4.875, .25, x_offset=-2.5, y_offset=2.5, rotation=0))
    panels[3].place_random_leds(leds_per_panel)
    panels[3].add_copy(2.5, 2.5, 90)
    panels[3].add_copy(2.5, -2.5, 180)
    panels[3].add_copy(-2.5, -2.5, 270)

    panels.append(panel(4.875, 4.875, .25, x_offset=-12.5, y_offset=7.5, rotation=0))
    panels[4].place_random_leds(leds_per_panel)
    panels[4].add_copy(12.5, 7.5, 90)
    panels[4].add_copy(12.5, -7.5, 180)
    panels[4].add_copy(-12.5, -7.5, 270)

    panels.append(panel(4.875, 4.875, .25, x_offset=-12.5, y_offset=2.5, rotation=0))
    panels[5].place_random_leds(leds_per_panel)
    panels[5].add_copy(12.5, 2.5, 180)
    panels[5].add_copy(12.5, -2.5, 180)
    panels[5].add_copy(-12.5, -2.5, 0)
    '''


    g = grow_area(22,22,3)
    for p in panels:
        g.add_panel(p)

    optimizer = flux_optimizer(g)
    for i in range(100):
        optimizer.optimizer_step(view_angle=view_angle)
        print(i)
    

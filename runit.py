import numpy as np
from mpmath import mp, mpf
from matplotlib import pyplot as plt

mp.dps = 50
LY = 9.461e15                    # metres per light-year
lightyear_convs = 1 / LY

ACCEL_COLOURS = {0.5: "blue", 1: "red", 1.5: "green", 3: "orange", 5: "purple"}


def colour_for(a):
    return ACCEL_COLOURS.get(a, "black")


def speed_str(gamma, extra=15):
    """Return v/c as 0.9999...xyz showing every leading nine plus `extra`
    more digits, together with 1 - v/c in scientific notation."""
    g = mpf(gamma)
    with mp.workdps(int(2 * float(mp.log10(g))) + extra + 60):
        gap = 1 / (g * (g + mp.sqrt(g ** 2 - 1)))    # exactly 1 - v/c, no cancellation
        u = 1 - gap
        nines = int(-mp.floor(mp.log10(gap)))
        return f"{mp.nstr(u, nines + extra)}c  (1-v/c = {mp.nstr(gap, 5)})"


def info_box(text):
    """Put an info box in the empty lower-right region of the current plot."""
    plt.gca().text(0.98, 0.02, text, transform=plt.gca().transAxes,
                   ha="right", va="bottom", fontsize=8,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))


def finish_plot(title, xmax):
    plt.plot([0, xmax], [0, xmax], "k--", label="Light ray")
    plt.xlabel("Space axis (light-years)")
    plt.ylabel("Time axis (ct, light-years)")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)


class Ship:

    def __init__(self, colour, acceleration):
        '''
        acceleration : acceleration of ship in g (multiples of 9.8 ms^-2)
        '''
        self.g = acceleration
        self.a = acceleration * 9.8
        self.c = 3e8
        self.rho = (self.c ** 2) / self.a          # metres
        self.rho_ly = self.rho * lightyear_convs   # light-years
        self.color = colour

    # ---------- plotting helpers (use the current axes) ----------

    def worldline(self, max_dist, label=None):
        x = np.linspace(0, max_dist, 500)                        # ly
        ct = np.sqrt(x * (x + 2 * self.rho_ly))                  # ly (same as sqrt((x+r)^2 - r^2))
        plt.plot(x, ct, color=self.color, label=label)

    def proper_time_marks(self, max_dist, tau_step=None, label=None):
        '''Dots on the worldline every tau_step years of astronaut time.'''
        r = self.rho_ly
        tau_max = r * np.arccosh(1 + max_dist / r)
        if tau_step is None:
            tau_step = float(f"{tau_max / 8:.1g}")   # roughly 8 dots
        taus = np.arange(tau_step, tau_max, tau_step)
        x = r * (np.cosh(taus / r) - 1)
        ct = r * np.sinh(taus / r)
        if label is True:
            label = f"Astronaut clock (dot every {tau_step:g} yr)"
        plt.plot(x, ct, "o", color=self.color, markersize=4, label=label)

    # ---------- the journeys ----------

    def fix_distance(self, max_dist, plot=True):
        '''One-way trip of max_dist light-years.'''
        r = self.rho_ly
        gamma = 1 + mpf(max_dist) / mpf(r)
        earth_years = float(r * mp.sqrt(gamma ** 2 - 1))
        tau_years = float(r * mp.acosh(gamma))

        info = (f"{self.g} g, one-way, {max_dist:g} ly\n"
                f"Earth time: {earth_years:.2f} yr\n"
                f"Astronaut time: {tau_years:.2f} yr\n"
                f"Final speed: {speed_str(gamma)}")
        print("\n" + info)

        if plot:
            plt.figure(figsize=(9, 6))
            self.worldline(max_dist, label=f"{self.g} g")
            self.proper_time_marks(max_dist, label=True)
            plt.plot(max_dist, earth_years, "k*", markersize=12, label="Arrival")
            info_box(info)
            finish_plot(f"One-way journey, fixed distance ({max_dist:g} ly)", max_dist)

    def fix_time(self, max_time, plot=True):
        '''One-way trip lasting max_time years of Earth time.'''
        r = self.rho_ly
        gamma = mp.sqrt(1 + (mpf(max_time) / mpf(r)) ** 2)
        dist_ly = float(r * (gamma - 1))
        tau_years = float(r * mp.asinh(mpf(max_time) / mpf(r)))

        info = (f"{self.g} g, one-way, {max_time:g} yr Earth time\n"
                f"Distance reached: {dist_ly:.2f} ly\n"
                f"Astronaut time: {tau_years:.2f} yr\n"
                f"Final speed: {speed_str(gamma)}")
        print("\n" + info)

        if plot:
            plt.figure(figsize=(9, 6))
            self.worldline(dist_ly, label=f"{self.g} g")
            self.proper_time_marks(dist_ly, label=True)
            plt.axhline(max_time, color="gray", linestyle=":", label=f"{max_time:g} yr Earth time")
            plt.plot(dist_ly, max_time, "k*", markersize=12, label="End of journey")
            info_box(info)
            finish_plot(f"One-way journey, fixed Earth time ({max_time:g} yr)", dist_ly)

    def fix_proper_time(self, tau_total, plot=True):
        '''
        One-way trip (constant acceleration throughout) lasting tau_total
        years of ASTRONAUT (proper) time. Same as fix_time, but the fixed
        quantity is the time on the ship's clock rather than Earth's.
        '''
        r = self.rho_ly
        gamma = mp.cosh(mpf(tau_total) / mpf(r))
        dist_ly = float(r * (gamma - 1))
        earth_years = float(r * mp.sinh(mpf(tau_total) / mpf(r)))

        info = (f"{self.g} g, one-way, {tau_total:g} yr astronaut time\n"
                f"Distance reached: {dist_ly:.2f} ly\n"
                f"Earth time: {earth_years:.2f} yr\n"
                f"Final speed: {speed_str(gamma)}")
        print("\n" + info)

        if plot:
            plt.figure(figsize=(9, 6))
            self.worldline(dist_ly, label=f"{self.g} g")
            self.proper_time_marks(dist_ly, label=True)
            plt.axhline(earth_years, color="gray", linestyle=":",
                        label=f"{earth_years:.2f} yr Earth time at end")
            plt.plot(dist_ly, earth_years, "k*", markersize=12,
                     label=f"End of journey (τ={tau_total:g} yr)")
            info_box(info)
            finish_plot(f"One-way journey, fixed astronaut time ({tau_total:g} yr)", dist_ly)

    def one_way_stop(self, dist, plot=True):
        '''
        One-way trip to a destination dist light-years away: accelerate to the
        halfway point, then decelerate to rest at the destination (no return).
        '''
        r = self.rho_ly
        h = dist / 2
        gamma = 1 + mpf(h) / mpf(r)                    # gamma at peak speed (cosh(tau/rho))
        tau_h = float(r * mp.acosh(gamma))             # proper time for one half-leg
        T_h = float(r * mp.sqrt(gamma ** 2 - 1))       # Earth time for one half-leg

        info = (f"{self.g} g, one-way with stop, {dist:g} ly\n"
                f"Earth time: {2 * T_h:.2f} yr\n"
                f"Astronaut time: {2 * tau_h:.2f} yr\n"
                f"Peak speed: {speed_str(gamma)} (at {tau_h:.2f} yr astronaut time)\n"
                f"Trip length contracted at peak speed: {dist / float(gamma):.3f} ly")
        print("\n" + info)

        if plot:
            tau = np.linspace(0, tau_h, 200)
            # leg 1: accelerate away, leg 2: decelerate to rest at destination
            x1 = r * (np.cosh(tau / r) - 1)
            t1 = r * np.sinh(tau / r)
            x2 = dist - r * (np.cosh((tau_h - tau) / r) - 1)
            t2 = 2 * T_h - r * np.sinh((tau_h - tau) / r)
            x = np.concatenate([x1, x2])
            t = np.concatenate([t1, t2])

            plt.figure(figsize=(9, 6))
            plt.plot(x, t, color=self.color, label=f"{self.g} g accelerate/decelerate")
            plt.axvline(0, color="gray", linestyle=":", label="Earth")
            plt.axvline(dist, color="gray", linestyle="--", label="Destination")
            plt.plot([0, dist], [0, 2 * T_h], "ko")
            plt.annotate(f"arrive, τ={2 * tau_h:.1f} yr", (dist, 2 * T_h),
                         textcoords="offset points", xytext=(-90, 5), fontsize=8)
            info_box(info)
            finish_plot(f"One-way journey with stop, {dist:g} ly", dist)

    def one_way_stop_fix_time(self, max_time, plot=True):
        '''
        One-way accelerate-then-decelerate trip (ending at rest) lasting
        max_time years of EARTH time in total. Works out the distance that
        can be covered, then runs one_way_stop for that distance.
        '''
        r = self.rho_ly
        T_h = mpf(max_time) / 2                                        # Earth time per half-leg
        dist = float(2 * r * (mp.sqrt(1 + (T_h / mpf(r)) ** 2) - 1))   # ly
        print(f"\n[Fixed Earth time: {max_time:g} yr -> distance covered: {dist:.4f} ly]")
        self.one_way_stop(dist, plot=plot)

    def one_way_stop_fix_proper_time(self, tau_total, plot=True):
        '''
        One-way accelerate-then-decelerate trip (ending at rest) lasting
        tau_total years of ASTRONAUT time in total. Works out the distance
        that can be covered, then runs one_way_stop for that distance.
        '''
        r = self.rho_ly
        tau_h = mpf(tau_total) / 2                                     # astronaut time per half-leg
        dist = float(2 * r * (mp.cosh(tau_h / mpf(r)) - 1))            # ly
        print(f"\n[Fixed astronaut time: {tau_total:g} yr -> distance covered: {dist:.4f} ly]")
        self.one_way_stop(dist, plot=plot)

    def return_journey(self, dist, plot=True):
        '''
        Round trip to a destination dist light-years away: accelerate to the
        halfway point, decelerate to rest at the destination, then repeat
        on the way back (no waiting time at the destination).
        '''
        r = self.rho_ly
        h = dist / 2
        gamma = 1 + mpf(h) / mpf(r)                    # gamma at peak speed (cosh(tau/rho))
        tau_h = float(r * mp.acosh(gamma))             # proper time for one half-leg
        T_h = float(r * mp.sqrt(gamma ** 2 - 1))       # Earth time for one half-leg

        info = (f"{self.g} g, round trip, {dist:g} ly each way\n"
                f"Total distance (Earth frame): {2 * dist:g} ly\n"
                f"Total Earth time: {4 * T_h:.2f} yr\n"
                f"Total astronaut time: {4 * tau_h:.2f} yr\n"
                f"Peak speed: {speed_str(gamma)} (at {tau_h:.2f} yr astronaut time)\n"
                f"Trip length contracted at peak speed: {dist / float(gamma):.3f} ly")
        print("\n" + info)

        if plot:
            tau = np.linspace(0, tau_h, 200)
            # leg 1: accelerate away, leg 2: decelerate to rest at destination
            x1 = r * (np.cosh(tau / r) - 1)
            t1 = r * np.sinh(tau / r)
            x2 = dist - r * (np.cosh((tau_h - tau) / r) - 1)
            t2 = 2 * T_h - r * np.sinh((tau_h - tau) / r)
            x_out = np.concatenate([x1, x2])
            t_out = np.concatenate([t1, t2])
            # return trip is the mirror image, starting at time 2*T_h
            x = np.concatenate([x_out, dist - x_out])
            t = np.concatenate([t_out, 2 * T_h + t_out])

            plt.figure(figsize=(9, 6))
            plt.plot(x, t, color=self.color, label=f"{self.g} g round trip")
            plt.axvline(0, color="gray", linestyle=":", label="Earth")
            plt.axvline(dist, color="gray", linestyle="--", label="Destination")
            plt.plot([0, dist, 0], [0, 2 * T_h, 4 * T_h], "ko")
            plt.annotate(f"arrive, τ={2 * tau_h:.1f} yr", (dist, 2 * T_h),
                         textcoords="offset points", xytext=(-90, 5), fontsize=8)
            plt.annotate(f"home, τ={4 * tau_h:.1f} yr", (0, 4 * T_h),
                         textcoords="offset points", xytext=(8, -4), fontsize=8)
            info_box(info)
            finish_plot(f"Round trip to {dist:g} ly", dist)

    def return_journey_fix_time(self, max_time, plot=True):
        '''
        Round trip (accelerate, decelerate, return, no waiting) lasting
        max_time years of EARTH time in total. Works out the one-way
        distance that can be reached, then runs return_journey for it.
        '''
        r = self.rho_ly
        T_h = mpf(max_time) / 4                                        # Earth time per half-leg
        dist = float(2 * r * (mp.sqrt(1 + (T_h / mpf(r)) ** 2) - 1))   # one-way ly
        print(f"\n[Fixed Earth time: {max_time:g} yr -> one-way distance: {dist:.4f} ly]")
        self.return_journey(dist, plot=plot)

    def return_journey_fix_proper_time(self, tau_total, plot=True):
        '''
        Round trip (accelerate, decelerate, return, no waiting) lasting
        tau_total years of ASTRONAUT time in total. Works out the one-way
        distance that can be reached, then runs return_journey for it.
        '''
        r = self.rho_ly
        tau_h = mpf(tau_total) / 4                                     # astronaut time per half-leg
        dist = float(2 * r * (mp.cosh(tau_h / mpf(r)) - 1))            # one-way ly
        print(f"\n[Fixed astronaut time: {tau_total:g} yr -> one-way distance: {dist:.4f} ly]")
        self.return_journey(dist, plot=plot)


# ---------- comparison plot ----------

def compare_accelerations(max_dist):
    plt.figure(figsize=(9, 6))
    for a, col in ACCEL_COLOURS.items():
        ship = Ship(col, a)
        ship.fix_distance(max_dist, plot=False)
        ship.worldline(max_dist, label=f"{a} g")
        ship.proper_time_marks(max_dist, tau_step=0.5)
    finish_plot("Worldlines for constant proper acceleration "
                "(dots every 0.5 yr astronaut time)", max_dist)


# ---------- interface ----------

MENU = """
=== Relativity Journey Simulator ===
 1  One-way journey: fixed distance
 2  One-way journey: fixed Earth time
 3  Round trip (accelerate, decelerate, return): fixed distance
 4  Compare worldlines for several accelerations
 5  One-way journey: accelerate then decelerate (no return), fixed distance
 6  One-way journey: fixed astronaut time
 7  Run everything
 8  Accelerate/decelerate one-way: fixed Earth time
 9  Accelerate/decelerate one-way: fixed astronaut time
10  Round trip: fixed Earth time
11  Round trip: fixed astronaut time
 q  Quit
"""


def ask(prompt, default):
    while True:
        raw = input(f"{prompt} [default {default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("Please enter a positive number.")


def main():
    while True:
        print(MENU)
        choice = input("> ").strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            a = ask("Acceleration (g)", 1)
            d = ask("Distance (light-years)", 10)
            Ship(colour_for(a), a).fix_distance(d)
        elif choice == "2":
            a = ask("Acceleration (g)", 1)
            t = ask("Earth time (years)", 10)
            Ship(colour_for(a), a).fix_time(t)
        elif choice == "3":
            a = ask("Acceleration (g)", 1)
            d = ask("One-way distance (light-years)", 10)
            Ship(colour_for(a), a).return_journey(d)
        elif choice == "4":
            d = ask("Max distance (light-years)", 10)
            compare_accelerations(d)
        elif choice == "5":
            a = ask("Acceleration (g)", 1)
            d = ask("Distance (light-years)", 10)
            Ship(colour_for(a), a).one_way_stop(d)
        elif choice == "6":
            a = ask("Acceleration (g)", 1)
            tau = ask("Astronaut time (years)", 10)
            Ship(colour_for(a), a).fix_proper_time(tau)
        elif choice == "7":
            a = ask("Acceleration (g)", 1)
            d = ask("Distance (light-years)", 10)
            t = ask("Earth time (years)", 10)
            tau = ask("Astronaut time (years)", 10)
            ship = Ship(colour_for(a), a)
            ship.fix_distance(d)
            ship.fix_time(t)
            ship.fix_proper_time(tau)
            ship.one_way_stop(d)
            ship.one_way_stop_fix_time(t)
            ship.one_way_stop_fix_proper_time(tau)
            ship.return_journey(d)
            ship.return_journey_fix_time(t)
            ship.return_journey_fix_proper_time(tau)
            compare_accelerations(d)
        elif choice == "8":
            a = ask("Acceleration (g)", 1)
            t = ask("Total Earth time (years)", 10)
            Ship(colour_for(a), a).one_way_stop_fix_time(t)
        elif choice == "9":
            a = ask("Acceleration (g)", 1)
            tau = ask("Total astronaut time (years)", 10)
            Ship(colour_for(a), a).one_way_stop_fix_proper_time(tau)
        elif choice == "10":
            a = ask("Acceleration (g)", 1)
            t = ask("Total Earth time (years)", 10)
            Ship(colour_for(a), a).return_journey_fix_time(t)
        elif choice == "11":
            a = ask("Acceleration (g)", 1)
            tau = ask("Total astronaut time (years)", 10)
            Ship(colour_for(a), a).return_journey_fix_proper_time(tau)
        else:
            print("Not a valid option, try again.")
            continue

        plt.show()   # blocks until you close the plot windows, then returns to the menu


if __name__ == "__main__":
    main()
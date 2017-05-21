Environment Parameters
======================

When using this MARS based environments there are some basic environmental
parameters, which can be changed within `learning_config.yml`:

* **velocityControl**: If `True` environment is loaded with *velocity
  controlled* motors. Otherwise *position controlled* motors are used.
* **calc_ms**: The time interval the physic engine does updates or the time
  between two physic updates.
* **stepTimeMs**: The time interval for one step of the environment. E.g. if
  *stepTimeMs = 20* and *calc_ms = 2* the environment will do *10* physic
  updates for each step.
* **graphicsUpdateTime**: The time interval for updating the graphical
  interface.
* **enableGUI**: If `False` MARS starts without GUI. Therefore the file
  `core_libs-nogui.txt` must exist.

Example code for the `Evironment Parameters` section in the
`learning_config.yml`:

    Environment Parameters:
      calc_ms: 10
      stepTimeMs: 10
      graphicsUpdateTime: 10
      velocityControl: True
      enableGUI: True

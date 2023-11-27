# -*- coding: utf-8 -*-
"""
/***************************************************************************
 AircraftLocalizationVis
                                 A QGIS plugin
 Visualize Sensor network and data from aircrafts
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-05-20
        copyright            : (C) 2023 by Iryna Ryabokin
        email                : riabokin.iryna1122@vu.cdu.edu.ua
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load AircraftLocalizationVis class from file AircraftLocalizationVis.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .aircraft_localization_vis import AircraftLocalizationVis
    return AircraftLocalizationVis(iface)

"""
    pyscm -- The Set Covering Machine in Python
    Copyright (C) 2017 Alexandre Drouin

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from six import iteritems


def _class_to_string(instance):
    """
    Returns a string representation of the public attributes of a class.

    Parameters:
    -----------
    instance: object
        An instance of any class.

    Returns:
    --------
    string_rep: string
        A string representation of the class and its public attributes.

    Notes:
    -----
    Private attributes must be marked with a leading underscore.
    """
    return (
        instance.__class__.__name__
        + "("
        + ",".join(
            [
                str(k) + "=" + str(v)
                for k, v in iteritems(instance.__dict__)
                if str(k[0]) != "_"
            ]
        )
        + ")"
    )

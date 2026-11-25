# Copyright 2026 Maxime SAUVAYRE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# "Commons Clause" License Condition v1.0
#
# The Software is provided to you by the Licensor under the License, as
# defined above, subject to the following condition.
#
# Without limiting other conditions in the License, the grant of rights under
# the License will not include, and the License does not grant to you, the
# right to Sell the Software.
#
# For purposes of the foregoing, "Sell" means practicing any or all of the
# rights granted to you under the License to provide to third parties, for a
# fee or other consideration (including without limitation fees for hosting
# or consulting/support services related to the Software), a product or
# service whose value derives, entirely or substantially, from the
# functionality of the Software. Any license notice or attribution required
# by the License must also include this Commons Clause License Condition
# notice.
#
# Software: picturesRC
# License: Apache License 2.0
# Licensor: Maxime SAUVAYRE

"""Cœur de traitement, sans aucune dépendance à l'interface graphique."""

from .straighten import TiltEstimate, estimate_tilt, inscribed_crop_size, measure_line_angle
from .crop import crop_is_valid, pixel_crop_box
from .imageio import load_image, load_preview, save_image
from .subject import PhotoAnalysis, SubjectEstimate, analyze_photo, detect_subject, estimate_outline_tilt
from .pipeline import ExportSettings, export_image

__all__ = [
    "TiltEstimate",
    "estimate_tilt",
    "inscribed_crop_size",
    "measure_line_angle",
    "crop_is_valid",
    "pixel_crop_box",
    "load_image",
    "load_preview",
    "save_image",
    "SubjectEstimate",
    "detect_subject",
    "estimate_outline_tilt",
    "analyze_photo",
    "PhotoAnalysis",
    "ExportSettings",
    "export_image",
]

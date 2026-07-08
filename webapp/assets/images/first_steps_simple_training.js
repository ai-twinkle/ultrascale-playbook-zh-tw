
// Function to enhance the SVG content by adding styles and data attributes
function enhanceSVGContent(originalContent) {
      const parser = new DOMParser();
      const doc = parser.parseFromString(originalContent, 'image/svg+xml');

      // Create a style element with hover effects and insert it as the first child of the SVG
      const styleElement = doc.createElementNS('http://www.w3.org/2000/svg', 'style');
      styleElement.textContent = `
      path[data-element-type="layer"] {
      transition: all 0.3s;
      cursor: pointer;
      }
      path[data-element-type="layer"]:hover {
      fill: #b197fc !important;
      transform: translate(0, -2px);
      }

      path[data-element-type="layer-updated"] {
      transition: all 0.3s;
      cursor: pointer;
      }
      
      path[data-element-type="layer-updated"]:hover {
      fill:rgb(103, 56, 244) !important;
      transform: scale(1.02);
      transform: translate(0, -2px);
      }

      path[data-element-type="gradient"] {
      transition: all 0.3s;
      cursor: pointer;
      }
      path[data-element-type="gradient"]:hover {
      fill: #f06595 !important;
      transform: translate(0, -2px);
      }

      path[data-element-type="forward"] {
      transition: all 0.3s;
      cursor: pointer;
      }
      path[data-element-type="forward"]:hover {
      stroke: #0c8599 !important;
      stroke-width: 4 !important;
      }

      path[data-element-type="backward"] {
      transition: all 0.3s;
      cursor: pointer;
      }
      path[data-element-type="backward"]:hover {
      stroke: #e8590c !important;
      stroke-width: 4 !important;
      }

      path[data-element-type="optimization"] {
      transition: all 0.3s;
      cursor: pointer;
      }
      path[data-element-type="optimization"]:hover {
      stroke: #087f5b !important;
      stroke-width: 4 !important;
      }
`;
      doc.documentElement.insertBefore(styleElement, doc.documentElement.firstChild);

      // Process neural network layers (purple nodes)
      doc.querySelectorAll('path[fill="#d0bfff"]').forEach((node, index) => {
            node.setAttribute('data-element-id', `layer-${index}`);
            node.setAttribute('data-element-type', 'layer');
      });

      doc.querySelectorAll('path[fill="#9775fa"]').forEach((node, index) => {
            node.setAttribute('data-element-id', `layer-updated-${index}`);
            node.setAttribute('data-element-type', 'layer-updated');
      });

      // Process gradient nodes (pink nodes)
      doc.querySelectorAll('path[fill="#f783ac"]').forEach((node, index) => {
            node.setAttribute('data-element-id', `gradient-${index}`);
            node.setAttribute('data-element-type', 'gradient');
      });

      // Process arrows by matching stroke colors
      const arrowTypes = {
            '#15aabf': 'forward',
            '#fd7e14': 'backward',
            '#099268': 'optimization'
      };

      Object.entries(arrowTypes).forEach(([color, type]) => {
            doc.querySelectorAll(`path[stroke="${color}"]`).forEach((arrow, index) => {
                  arrow.setAttribute('data-element-id', `${type}-${index}`);
                  arrow.setAttribute('data-element-type', type);
            });
      });

      // Make the SVG responsive
      doc.documentElement.setAttribute('width', '100%');
      doc.documentElement.setAttribute('height', '100%');
      doc.documentElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');

      return new XMLSerializer().serializeToString(doc);
}

// Function to load an SVG file via fetch
async function loadSVG(url, containerId) {
      try {
            const response = await fetch(url);
            if (!response.ok) {
                  throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const svgText = await response.text();
            const enhancedSVG = enhanceSVGContent(svgText);
            document.getElementById(containerId).innerHTML = enhancedSVG;
      } catch (error) {
            console.error('Error loading SVG:', error);
            document.getElementById(containerId).innerHTML = '<p>Error loading SVG.</p>';
      }
}

// Load the SVG file (adjust the path if needed)
loadSVG('../assets/images/first_steps_simple_training.svg', 'svg-first_steps_simple_training');

// Set up event listeners to display a description of the hovered element
const svgContainer = document.getElementById('svg-first_steps_simple_training');

svgContainer.addEventListener('mouseover', function (event) {
      const target = event.target;
      if (target.tagName.toLowerCase() === 'path' && target.hasAttribute('data-element-id')) {
            const elementId = target.getAttribute('data-element-id');
            const elementType = target.getAttribute('data-element-type');
            const descriptions = {
                  layer: 'Neural Network Layer',
                  'layer-updated': 'Neural Network Layer (updated)',
                  gradient: 'Gradient Update Layer',
                  forward: 'Forward Pass Connection',
                  backward: 'Backward Pass Connection',
                  optimization: 'Optimization Step'
            };
            const description = descriptions[elementType] || elementType;
            document.getElementById('svg-first_steps_simple_training-info').textContent = `Hovering over: ${description} (${elementId})`;
      }
});

svgContainer.addEventListener('mouseout', function () {
      document.getElementById('svg-first_steps_simple_training-info').textContent = 'Hover over the network elements to see their details';
});

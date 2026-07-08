// first_steps_memory_profile.js

// Function to enhance the SVG content by adding styles and data attributes
function enhanceSVGContent2(originalContent) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(originalContent, 'image/svg+xml');

  // Create a style element with hover effects and insert it as the first child of the SVG
  const styleElement = doc.createElementNS('http://www.w3.org/2000/svg', 'style');
  styleElement.textContent = `
    /* Memory Block (free memory) */
    path[data-element-type="memory-block"] {
      transition: all 0.3s;
      cursor: pointer;
    }
    path[data-element-type="memory-block"]:hover {
      fill: #a5d6a7 !important; /* slightly darker than original */
      transform: translate(0, -2px);
    }

    /* Memory Block (updated) */
    path[data-element-type="memory-block-updated"] {
      transition: all 0.3s;
      cursor: pointer;
    }
    path[data-element-type="memory-block-updated"]:hover {
      fill: #81c784 !important;
      transform: scale(1.02) translate(0, -2px);
    }

    /* Stack Segment */
    path[data-element-type="stack"] {
      transition: all 0.3s;
      cursor: pointer;
    }
    path[data-element-type="stack"]:hover {
      fill: #ffd54f !important;
      transform: translate(0, -2px);
    }

    /* Read Operation Arrow */
    path[data-element-type="read"] {
      transition: all 0.3s;
      cursor: pointer;
    }
    path[data-element-type="read"]:hover {
      stroke: #1e88e5 !important;
      stroke-width: 4 !important;
    }

    /* Write Operation Arrow */
    path[data-element-type="write"] {
      transition: all 0.3s;
      cursor: pointer;
    }
    path[data-element-type="write"]:hover {
      stroke: #d32f2f !important;
      stroke-width: 4 !important;
    }

    /* Cache Operation Arrow */
    path[data-element-type="cache"] {
      transition: all 0.3s;
      cursor: pointer;
    }
    path[data-element-type="cache"]:hover {
      stroke: #fbc02d !important;
      stroke-width: 4 !important;
    }
  `;
  doc.documentElement.insertBefore(styleElement, doc.documentElement.firstChild);

  // Process Memory Blocks (free memory segments) – assume they have fill color "#c8e6c9"
  doc.querySelectorAll('path[fill="#c8e6c9"]').forEach((node, index) => {
    node.setAttribute('data-element-id', `memory-block-${index}`);
    node.setAttribute('data-element-type', 'memory-block');
  });

  // Process Updated Memory Blocks – assume they have fill color "#a5d6a7"
  doc.querySelectorAll('path[fill="#a5d6a7"]').forEach((node, index) => {
    node.setAttribute('data-element-id', `memory-block-updated-${index}`);
    node.setAttribute('data-element-type', 'memory-block-updated');
  });

  // Process Stack Segments – assume they have fill color "#ffe082"
  doc.querySelectorAll('path[fill="#ffe082"]').forEach((node, index) => {
    node.setAttribute('data-element-id', `stack-${index}`);
    node.setAttribute('data-element-type', 'stack');
  });

  // Process arrows for memory operations based on stroke colors
  // Assume: 
  //   stroke "#42a5f5" for read operations,
  //   stroke "#ef5350" for write operations,
  //   stroke "#ffca28" for cache operations.
  const arrowTypes = {
    '#42a5f5': 'read',
    '#ef5350': 'write',
    '#ffca28': 'cache'
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
    console.log('Loading SVG from:', url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const svgText = await response.text();
    const enhancedSVG = enhanceSVGContent2(svgText);
    document.getElementById(containerId).innerHTML = enhancedSVG;
  } catch (error) {
    console.error('Error loading SVG:', error);
    document.getElementById(containerId).innerHTML = '<p>Error loading SVG.</p>';
  }
}

// Load the SVG file (adjust the path if needed)
loadSVG('../assets/images/first_steps_memory_profile.svg', 'svg-first_steps_memory_profile');

// Set up event listeners to display a description of the hovered element
const svgContainer2 = document.getElementById('svg-first_steps_memory_profile');

svgContainer2.addEventListener('mouseover', function(event) {
  const target = event.target;
  if (target.tagName.toLowerCase() === 'path' && target.hasAttribute('data-element-id')) {
    const elementId = target.getAttribute('data-element-id');
    const elementType = target.getAttribute('data-element-type');
    const descriptions = {
      'memory-block': 'Memory Block',
      'memory-block-updated': 'Memory Block (updated)',
      'stack': 'Stack Segment',
      'read': 'Memory Read',
      'write': 'Memory Write',
      'cache': 'Cache Operation'
    };
    const description = descriptions[elementType] || elementType;
    document.getElementById('svg-first_steps_memory_profile-info').textContent = `Hovering over: ${description} (${elementId})`;
  }
});

svgContainer2.addEventListener('mouseout', function() {
  document.getElementById('svg-first_steps_memory_profile-info').textContent = 'Hover over the elements to see their details';
});

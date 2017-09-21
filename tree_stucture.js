/* HTML Format

<body>
	<div>
		<a></a>
		<p></p>
	</div>
</body>

*/


// Tree Format

[ // new el
  [1, 0, 0, 0], // tagname='body'
  [ // body's children
    [ // new el
      [0, 1, 0, 0], // tagname='div'
      [ // div's children
        [ // new el
          [0, 0, 1, 0], // tagname='a'
          [] // a's children
        ],
        [ // new el
          [0, 0, 0, 1], // tagname='p'
          [] // p's children
        ]
      ]
    ]
  ]
]


// Storage Format

/* We have an available tagnames_map (or classes map):
{
	'body': 0,
	'div': 1,
	'a': 2,
	'p': 3,
	'null': -1
}	
*/

// Max Elements: 201 (the 1 being <body>)
// Max Children/Element: 10
// Max Nodes: 2001
// Max Depth: 20

classForNodeIndex = [
  0,  // body (div + 9 null children)
  1,  // div (a + p + 8 null children)
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  2,  // a
  3,  // p
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1, // null
  -1 // null
    // ...
]

nodeIndexConnectionsForNodeIndex = [
  /*  0  */[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // node 0 holds nodes 1-10
  /*  1  */[11, 12, 13, 14, 15, 16, 17, 18, 19, 20], // node 1 holds nodes 11-20
  /*  2  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], // no other nodes have any children
  /*  3  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  4  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  5  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  6  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  7  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  8  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  9  */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  10 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  11 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  12 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  13 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  14 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  15 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  16 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  17 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  18 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  19 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /*  20 */[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  /* ... */
  /* 2000*/[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
]
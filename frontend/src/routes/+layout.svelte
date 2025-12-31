<script>
	import '../app.css';
	
	// Navigation items for the sidebar
		const navItems = [
		{ name: 'Home', path: '/', icon: 'home' },
		{ 
			name: 'API Testing', 
			icon: 'api',
			children: [
				{ name: 'METAR', path: '/api/metar', description: 'Current airport weather' },
				{ name: 'TAF', path: '/api/taf', description: 'Terminal aerodrome forecasts' },
				{ name: 'PIREP', path: '/api/pirep', description: 'Pilot weather reports' },
				{ name: 'SIGMET/AIRMET', path: '/api/sigmet', description: 'Weather advisories' }
			]
		},
		{ name: 'Airport Summary', path: '/airport-summary', icon: 'summary', description: 'AI-powered comprehensive weather analysis' },
		{ name: 'METAR Display', path: '/cockpit-metar2', icon: 'display', description: 'Pilot-friendly weather display' },
		{ name: 'TAF Display', path: '/cockpit-taf', icon: 'forecast', description: 'Interactive forecast visualization' },
		{ name: 'API Documentation', path: '/api/docs', icon: 'document' }
	];

	let { children } = $props();
	let isSidebarOpen = $state(false);
	let station = ''; // Add initialization for station variable

	// Toggle sidebar on mobile
	function toggleSidebar() {
		isSidebarOpen = !isSidebarOpen;
	}

	// Icon components
	const icons = {
		home: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />`,
		api: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />`,
		document: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />`,
		menu: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />`,
		close: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />`,
		display: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />`,
		forecast: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />`,
		summary: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />`
	};
</script>

<!-- Main layout with sidebar and content -->
<div class="min-h-screen bg-gray-50 flex">
	<!-- Mobile sidebar toggle -->
	<div class="lg:hidden fixed top-0 left-0 z-20 p-4">
		<button
			onclick={toggleSidebar}
			class="p-2 rounded-md text-gray-500 hover:text-gray-800 hover:bg-gray-100 focus:outline-none"
		>
			<svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
				{@html icons.menu}
			</svg>
		</button>
	</div>

	<!-- Sidebar backdrop (mobile only) -->
	{#if isSidebarOpen}
		<button 
			class="lg:hidden fixed inset-0 bg-gray-600 bg-opacity-75 z-10"
			onclick={toggleSidebar}
			aria-label="Close sidebar"
			onkeydown={(e) => e.key === 'Escape' && toggleSidebar()}
		></button>
	{/if}

	<!-- Sidebar -->
	<aside
		class={`fixed inset-y-0 left-0 z-20 w-64 transform transition-transform duration-300 ease-in-out bg-white border-r border-gray-200 lg:translate-x-0 ${
			isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
		}`}
	>
		<div class="flex items-center justify-between h-16 px-4 border-b border-gray-200">
			<div class="flex items-center">
				<span class="text-lg font-semibold text-blue-600">AVIATION INSIGHT</span>
			</div>
			<button
				class="lg:hidden p-1 text-gray-500 hover:text-gray-800 focus:outline-none"
				onclick={toggleSidebar}
				aria-label="Close sidebar"
			>
				<svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					{@html icons.close}
				</svg>
			</button>
		</div>
		
		<div class="px-2 py-4 overflow-y-auto h-full">
			<nav class="space-y-6">
				{#each navItems as item}
					{#if item.children}
						<div>
							<div class="flex items-center px-3 py-2 text-sm font-medium text-gray-600">
								<svg class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									{@html icons[item.icon]}
								</svg>
								{item.name}
							</div>
							<div class="mt-1 ml-6 space-y-1">
								{#each item.children as child}
									<a
										href={child.path}
										class="block px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-100 hover:text-gray-900"
									>
										{child.name}
										<span class="block text-xs text-gray-500">{child.description}</span>
									</a>
								{/each}
							</div>
						</div>
					{:else}
						<a
							href={item.path}
							class="flex items-center px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-100 hover:text-gray-900"
						>
							<svg class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								{@html icons[item.icon]}
							</svg>
							{item.name}
						</a>
					{/if}
				{/each}
			</nav>
		</div>
	</aside>

	<!-- Main content -->
	<main class="flex-1 overflow-auto lg:ml-64">
		<div class="py-6 px-4 sm:px-6 lg:px-8">
			{@render children()}
		</div>
	</main>
</div>

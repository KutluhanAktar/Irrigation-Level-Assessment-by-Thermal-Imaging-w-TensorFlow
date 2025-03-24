<?php

// Insert a new row to the given CSV file:
function insert_new_line($csv_file, $line){
	$line = array($line);
	$f = fopen($csv_file.".csv", "a");
	fputcsv($f, explode(",", $line[0]));
	fclose($f);
	echo "The given line is added to the <i><b>".$csv_file.".csv</b></i> file successfully!";
}

// Create a new CSV file with the given name:
function create_csv_file($csv_file){
    $f = fopen($csv_file.'.csv', 'wb');
    fclose($f);
	echo "<i><b>".$csv_file.".csv</b></i> file created successfully!<br><br>";
}

// Create the required CSV files if requested.
if(isset($_GET["create_files"]) && $_GET["create_files"] == "ok"){
	create_csv_file("excessive"); create_csv_file("sufficient"); create_csv_file("moderate"); create_csv_file("dry");	
}

if(isset($_GET["thermal_img"]) && isset($_GET["level"])){
	insert_new_line($_GET["level"], $_GET["thermal_img"]);
}else{
	echo "Waiting Data...";
}

?>
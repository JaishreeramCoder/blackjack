import React, { useState, useEffect } from 'react';
import background from '../../assets/background.jpg'; // Updated background image path

const cardValues = [
    "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"
];

const CardModal = ({
    title,
    onCardSelect,
    onClose,
    isCardDisabled = () => false,
    allowClose = false
}) => {
    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div
                className="w-11/12 modal-md-half rounded-2xl shadow-2xl overflow-hidden relative"
                style={{
                    backgroundImage: `url(${background})`,
                    backgroundSize: "cover",
                    backgroundPosition: "center"
                }}
            >
                {/* Overlay for legibility */}
                <div className="absolute inset-0 bg-black opacity-50"></div>
                <div className="relative p-6">
                    {/* Modal Heading */}
                    <h2 className="text-2xl font-bold text-white text-center drop-shadow">
                        {title}
                    </h2>
                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                        {cardValues.map((card, index) => (
                            <button
                                key={index}
                                onClick={() => onCardSelect(card)}
                                disabled={isCardDisabled(card)}
                                className={`py-4 px-6 rounded-lg transition transform duration-200 ${
                                    isCardDisabled(card)
                                        ? "bg-gray-400 text-white cursor-not-allowed"
                                        : "bg-blue-500 text-white hover:shadow-2xl cursor-pointer hover:scale-105"
                                }`}
                            >
                                {card}
                            </button>
                        ))}
                    </div>
                    {allowClose && (
                        <div className="relative p-4 flex justify-end">
                            <button
                                onClick={onClose}
                                className="text-white hover:text-blue-500 cursor-pointer font-medium"
                            >
                                Close
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

const BlackjackGame = () => {
    const [gameStarted, setGameStarted] = useState(false);
    const [dealerCard, setDealerCard] = useState(null);
    const [playerCards, setPlayerCards] = useState([]);
    const [showDealerModal, setShowDealerModal] = useState(false);
    const [showPlayerFirstModal, setShowPlayerFirstModal] = useState(false);
    const [showPlayerSecondModal, setShowPlayerSecondModal] = useState(false);
    const [actionMessage, setActionMessage] = useState('');
    const [disableActions, setDisableActions] = useState(false);
    const [showNewCardModal, setShowNewCardModal] = useState(false);
    const [currentAction, setCurrentAction] = useState(null); // "hit" or "double"
    const [isFirstMove, setIsFirstMove] = useState(true);

    // Helper function to convert a card string to its numeric value (ace counted as 1)
    const getCardValue = (card) => {
        if (card === "A") return 1;
        if (["J", "Q", "K"].includes(card)) return 10;
        return parseInt(card, 10);
    };

    // Compute the state: [player_sum, usable_ace, dealer_card, is_first_move, cnt_A, cnt_2, ..., cnt_9, cnt_10group]
    const computeState = () => {
        let playerSum = 0;
        let usableAce = 0;
        let counts = {
            "A": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
            "7": 0,
            "8": 0,
            "9": 0,
            "10group": 0 // counts 10, J, Q, K
        };

        playerCards.forEach(card => {
            if (card === "A") {
                counts["A"] += 1;
            } else if (["J", "Q", "K", "10"].includes(card)) {
                counts["10group"] += 1;
            } else {
                counts[card] = (counts[card] || 0) + 1;
            }
            playerSum += getCardValue(card);
        });

        if (playerCards.includes("A") && playerSum + 10 <= 21) {
            usableAce = 1;
            playerSum += 10;
        }

        const dealerValue = getCardValue(dealerCard);

        return [
            playerSum,
            usableAce,
            dealerValue,
            isFirstMove ? 1 : 0, // New parameter: is_first_move
            counts["A"],
            counts["2"] || 0,
            counts["3"] || 0,
            counts["4"] || 0,
            counts["5"] || 0,
            counts["6"] || 0,
            counts["7"] || 0,
            counts["8"] || 0,
            counts["9"] || 0,
            counts["10group"]
        ];
    };

    // Fetch best action when dealer and initial two cards are selected
    useEffect(() => {
        if (disableActions) return;
    
        if (dealerCard && playerCards.length >= 2) {
            const stateArray = computeState();
            const playerSum = stateArray[0];
    
            if (playerSum > 21) {
                setActionMessage("You have lost the game because busted.");
                setDisableActions(true);
            } else if (playerSum === 21) {
                setActionMessage("You have got natural blackjack.");
                setDisableActions(true);
            } else {
                const requestOptions = {
                    method: "POST",
                    mode: "cors",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ state: stateArray })
                };
    
                const fetchFromUrl = (url) => {
                    return fetch(url, requestOptions).then(response => {
                        if (!response.ok) {
                            throw new Error(`Server error: ${response.statusText}`);
                        }
                        return response.json();
                    });
                };
    
                fetchFromUrl("http://localhost:8000/advisor/predict/")
                    .catch(localError => {
                        console.error("Error fetching from localhost:", localError);
                        // Fallback to alternative URL
                        return fetchFromUrl("https://blackjack-awh6.onrender.com/advisor/predict/");
                    })
                    .then(data => {
                        const action = data.action;
                        let actionStr = "";
                        switch (action) {
                            case 0:
                                actionStr = "Hit";
                                break;
                            case 1:
                                actionStr = "Stand";
                                break;
                            case 2:
                                actionStr = "Double Down";
                                break;
                            case 3:
                                actionStr = "Surrender";
                                break;
                            default:
                                actionStr = "Unknown Action";
                        }
                        setActionMessage(`You should choose ${actionStr} for maximum rewards.`);
                    })
                    .catch(error => {
                        console.error("Error fetching best action:", error);
                        setActionMessage("Error fetching action.");
                    });
            }
        }
    }, [dealerCard, playerCards, disableActions]);
    

    const startGame = () => {
        setDealerCard(null);
        setPlayerCards([]);
        setActionMessage('');
        setDisableActions(false);
        setIsFirstMove(true);
        setCurrentAction(null);
        setGameStarted(true);
        setShowDealerModal(true);
    };

    const handleDealerCardSelect = (card) => {
        setDealerCard(card);
        setShowDealerModal(false);
        setShowPlayerFirstModal(true);
    };

    const handlePlayerFirstCardSelect = (card) => {
        setPlayerCards([card]);
        setShowPlayerFirstModal(false);
        setShowPlayerSecondModal(true);
    };

    const handlePlayerSecondCardSelect = (card) => {
        setPlayerCards((prev) => [...prev, card]);
        setShowPlayerSecondModal(false);
    };

    // For Hit and Double Down actions that require a card selection modal
    const handleActionWithCard = (actionType) => {
        setCurrentAction(actionType);
        setShowNewCardModal(true);
    };

    // For actions that do not require a card selection modal
    const handleStandClick = () => {
        setActionMessage("Stand chosen. No further actions allowed.");
        setDisableActions(true);
    };

    const handleSurrenderClick = () => {
        setActionMessage("Surrender chosen. No further actions allowed.");
        setDisableActions(true);
    };

    // Called when a new card is selected from the modal (for Hit or Double Down)
    const handleNewCardSelect = (card) => {
        setPlayerCards((prev) => [...prev, card]);
        setShowNewCardModal(false);
        if (currentAction === "double") {
            setActionMessage("Double Down chosen. No further actions allowed.");
            setDisableActions(true);
        }
        setCurrentAction(null);
        setIsFirstMove(false);
    };

    // Check if a card has been selected 4 times already
    const isCardDisabled = (card) => {
        const count = playerCards.filter(c => c === card).length;
        return count >= 4;
    };

    return (
        <div
            className="min-h-screen bg-cover bg-center flex flex-col items-center justify-center p-4"
            style={{
                backgroundImage: `url(${background})`
            }}
        >
            {/* If game not started, show centered heading and start button */}
            {!gameStarted && (
                <div className="text-center">
                    <h1 className="text-5xl font-extrabold text-white drop-shadow-lg mb-8">
                        Blackjack Game
                    </h1>
                    <button
                        onClick={startGame}
                        className="bg-blue-600 text-white py-4 px-8 rounded-md text-xl shadow-xl hover:bg-blue-700 && cursor-pointer transform hover:scale-105 transition-transform"
                    >
                        Start Game
                    </button>
                </div>
            )}

            {/* Once game is started, display the game interface */}
            {gameStarted && (
                <>
                    {/* Row 1: Dealer's Card */}
                    {dealerCard && (
                        <div className="mb-6 text-center">
                            <h2 className="font-bold text-white text-2xl">Dealer's Card:</h2>
                            <div className="bg-white border border-gray-300 rounded-2xl p-6 shadow-xl transition transform hover:scale-105 inline-block text-3xl font-semibold">
                                {dealerCard}
                            </div>
                        </div>
                    )}

                    {/* Row 2: Player's Cards */}
                    {playerCards.length > 0 && (
                        <div className="mb-6 text-center">
                            <h2 className="font-bold text-white text-2xl">Player's Cards:</h2>
                            <div className="flex flex-wrap justify-center gap-4">
                                {playerCards.map((card, index) => (
                                    <div
                                        key={index}
                                        className="bg-white border border-gray-300 rounded-2xl p-6 shadow-xl transition transform hover:scale-105 text-3xl font-semibold"
                                    >
                                        {card}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Row 3: Action Message */}
                    {actionMessage && (
                        <div className="mb-6 text-center">
                            <h2 className="text-2xl font-semibold text-white drop-shadow-lg">
                                {actionMessage}
                            </h2>
                        </div>
                    )}

                    {/* Row 4: Action Buttons */}
                    {actionMessage && playerCards.length >= 2 && !disableActions && (
                        <div className="mb-6 grid grid-cols-2 sm:grid-cols-2 grid-cols-md-4 gap-4 w-full max-w-xl">
                            <button
                                onClick={() => handleActionWithCard("hit")}
                                className="bg-blue-500 text-white py-3 px-6 rounded hover:bg-blue-600 cursor-pointer transition-colors text-lg transform hover:scale-105"
                            >
                                Hit
                            </button>
                            <button
                                onClick={handleStandClick}
                                className="bg-blue-500 text-white py-3 px-6 rounded hover:bg-blue-600 cursor-pointer transition-colors text-lg transform hover:scale-105"
                            >
                                Stand
                            </button>
                            <button
                                disabled={!isFirstMove}
                                onClick={() => handleActionWithCard("double")}
                                className={`bg-blue-500 text-white py-3 px-6 rounded hover:bg-blue-600 cursor-pointer transition-colors text-lg transform hover:scale-105 ${
                                    !isFirstMove ? "opacity-50 cursor-not-allowed" : ""
                                }`}
                            >
                                Double Down
                            </button>
                            <button
                                disabled={!isFirstMove}
                                onClick={handleSurrenderClick}
                                className={`bg-blue-500 text-white py-3 px-6 rounded hover:bg-blue-600 cursor-pointer transition-colors text-lg transform hover:scale-105 ${
                                    !isFirstMove ? "opacity-50 cursor-not-allowed" : ""
                                }`}
                            >
                                Surrender
                            </button>
                        </div>
                    )}


                    {/* Restart Button placed below action buttons */}
                    <button
                        onClick={startGame}
                        className="bg-blue-500 text-white py-4 px-8 rounded-md text-xl shadow-xl hover:bg-blue-600 && cursor-pointer transform hover:scale-105 transition-transform mt-4"
                    >
                        Restart Game
                    </button>
                </>
            )}

            {/* Modal for Dealer's Card (Close disabled until selection) */}
            {showDealerModal && (
                <CardModal
                    title="What was the dealer's card?"
                    onCardSelect={handleDealerCardSelect}
                    onClose={() => {}}
                    allowClose={false}
                />
            )}

            {/* Modal for Player's First Card (Close disabled until selection) */}
            {showPlayerFirstModal && (
                <CardModal
                    title="What was your first card?"
                    onCardSelect={handlePlayerFirstCardSelect}
                    onClose={() => {}}
                    allowClose={false}
                />
            )}

            {/* Modal for Player's Second Card (Close disabled until selection) */}
            {showPlayerSecondModal && (
                <CardModal
                    title="What is your second card?"
                    onCardSelect={handlePlayerSecondCardSelect}
                    onClose={() => {}}
                    allowClose={false}
                />
            )}

            {/* Modal for New Card when Hit or Double Down is selected (Close allowed) */}
            {showNewCardModal && (
                <CardModal
                    title="What was the card you got?"
                    onCardSelect={handleNewCardSelect}
                    onClose={() => setShowNewCardModal(false)}
                    isCardDisabled={isCardDisabled}
                    allowClose={true}
                />
            )}
        </div>
    );
};

export default BlackjackGame;